"""Career-advisor tools. CONTRACT (implementation: agent L1; used by the agent graph: agent L2).

The career advisor is the ONE autonomous, tool-using agent of the project: unlike the rest of
the pipeline (deterministic workflow), the LLM decides which tools to call, in which order and
when to stop (ReAct loop). All tools are LOCAL: no network access at runtime.

Plain async functions (wrapped as LangChain tools by the agent graph). Each returns JSON-able
data, never raises for "not found" (returns an empty result + a short `note` instead), and
keeps outputs short (they are fed back to the LLM).

    async def search_skill_taxonomy(query: str, language: str = "en", limit: int = 5) -> dict
        Fuzzy search in the local ESCO snapshot (data/career/esco_skills.json).
        -> {"results": [{"uri", "preferred_label", "alt_labels"[:5], "description"[:300],
                         "related": [{"uri", "label"}][:8]}], "note": str}

    async def check_cv_evidence(skill: str, cv_text: str) -> dict
        Calibrated yes/no via the decision layer (jobmatch.decisions.service.check_skill_evidence).
        cv_text is injected by the graph, NOT chosen by the LLM.
        -> {"skill", "p_evidence": float | None, "present": bool | None, "note": str}

    async def search_learning_resources(skill: str, limit: int = 3) -> dict
        Lookup in the curated local catalog (data/career/learning_resources.json).
        -> {"results": [{"title", "url", "provider", "kind", "free": bool, "level",
                         "language", "est_hours"}], "note": str}
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import cache
from pathlib import Path
from typing import Any

from jobmatch.config import get_settings

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]  # src/jobmatch/career/tools.py -> repo
ESCO_FILE = "esco_skills.json"
RESOURCES_FILE = "learning_resources.json"

MAX_LIMIT = 10
MAX_ALT, MAX_DESC, MAX_RELATED = 5, 300, 8
MIN_SCORE = 0.6  # below this a candidate is not a match
MAX_SUFFIX = 3  # "postgres" matches "postgresql", "excel" does not match "excellence"
FUZZY_MIN_RATIO = 0.78  # difflib ratio needed for a typo-level match ("Pyhton")
KIND_ORDER = {"tutorial": 0, "course": 1, "docs": 2, "certification": 3}

__all__ = ["check_cv_evidence", "search_learning_resources", "search_skill_taxonomy"]


# --- text normalisation & scoring -----------------------------------------------------------


def normalize(text: str) -> str:
    """casefold + strip accents + collapse punctuation (keeps + # . for C++, C#, Next.js)."""
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9+#.]+", " ", text)
    return re.sub(r"\s+", " ", text.replace(". ", " ")).strip(" .")


def _tokens(text: str) -> frozenset[str]:
    return frozenset(t for t in re.split(r"[ .]+", text) if t)


def _strip_qualifier(label: str) -> str:
    """'python (computer programming)' -> 'python'."""
    return re.sub(r"\s*\(.*\)\s*$", "", label).strip()


def _score(query: str, q_tokens: frozenset[str], label: str, weight: float = 1.0) -> float:
    """Match quality of a normalised query against one normalised label, in [0, 1]."""
    if not query or not label:
        return 0.0
    if query == label:
        return weight
    if query == _strip_qualifier(label):
        return 0.98 * weight
    l_tokens = _tokens(label)
    if q_tokens and q_tokens <= l_tokens:  # every query word in the label: prefer short labels
        return (0.8 + 0.15 * len(query) / len(label)) * weight
    if l_tokens and l_tokens <= q_tokens:  # label is contained in the query ("sql queries")
        return (0.65 + 0.15 * len(label) / len(query)) * weight
    if len(query) >= 4 and any(  # truncated word ("postgres" -> "postgresql"), not "excel..."
        t.startswith(query) and len(t) - len(query) <= MAX_SUFFIX for t in l_tokens
    ):
        return 0.7 * weight
    ratio = SequenceMatcher(None, query, label).ratio()
    return 0.8 * ratio * weight if ratio >= FUZZY_MIN_RATIO else 0.0


def _best(query: str, labels: list[tuple[str, float]]) -> float:
    q_tokens = _tokens(query)
    return max((_score(query, q_tokens, lab, w) for lab, w in labels), default=0.0)


def _clip_limit(limit: int) -> int:
    try:
        return max(1, min(int(limit), MAX_LIMIT))
    except (TypeError, ValueError):
        return 3


# --- data loading (lazy, once per directory) ------------------------------------------------


def data_dir() -> Path:
    """`career_data_dir` setting (env CAREER_DATA_DIR); a relative path is resolved against
    the working directory first (Docker: WORKDIR /app), then the repository root."""
    configured = Path(get_settings().career_data_dir).expanduser()
    if configured.is_absolute():
        return configured
    for base in (Path.cwd(), REPO_ROOT):
        if (base / configured).is_dir():
            return base / configured
    return REPO_ROOT / configured


@dataclass(frozen=True)
class _Entry:
    labels: list[tuple[str, float]]  # (normalised label, weight)
    data: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("career data unavailable: %s (%s)", path, exc)
        return {}


@cache
def _esco_index(path: Path) -> list[_Entry]:
    raw = _read_json(path)
    seed_alias: dict[str, list[str]] = {}
    for seed, uris in (raw.get("seeds") or {}).items():
        if uris:  # the seed text is an alias of its best ESCO hit ("AWS" never maps: not found)
            seed_alias.setdefault(uris[0], []).append(seed)
    entries = []
    for s in raw.get("skills") or []:
        pref, alt = s.get("preferred_label") or {}, s.get("alt_labels") or {}
        labels = [(normalize(v), 1.0) for v in pref.values() if v]
        labels += [(normalize(a), 0.95) for lang in ("en", "fr") for a in alt.get(lang) or []]
        labels += [(normalize(a), 0.95) for a in seed_alias.get(s["uri"], [])]
        entries.append(_Entry(labels=list(dict.fromkeys(labels)), data=s))
    return entries


@cache
def _resource_index(path: Path) -> list[_Entry]:
    raw = _read_json(path)
    entries = []
    for r in raw.get("resources") or []:
        labels = [(normalize(r["skill"]), 1.0)]
        labels += [(normalize(a), 0.95) for a in r.get("aliases") or []]
        entries.append(_Entry(labels=labels, data=r))
    return entries


# --- tools ----------------------------------------------------------------------------------


async def search_skill_taxonomy(query: str, language: str = "en", limit: int = 5) -> dict:
    """Fuzzy search (en + fr labels, accent-insensitive) in the local ESCO snapshot."""
    q = normalize(query or "")
    if not q:
        return {"results": [], "note": "empty query"}
    index = _esco_index(data_dir() / ESCO_FILE)
    if not index:
        return {"results": [], "note": "ESCO snapshot unavailable"}
    lang = "fr" if (language or "").lower().startswith("fr") else "en"
    scored = sorted(
        ((score, e) for e in index if (score := _best(q, e.labels)) >= MIN_SCORE),
        key=lambda x: (-x[0], len(x[1].data["preferred_label"]["en"])),
    )[: _clip_limit(limit)]
    results = []
    for _, e in scored:
        s = e.data
        alt = s["alt_labels"].get(lang) or s["alt_labels"].get("en") or []
        results.append(
            {
                "uri": s["uri"],
                "preferred_label": s["preferred_label"].get(lang) or s["preferred_label"]["en"],
                "alt_labels": alt[:MAX_ALT],
                "description": (s.get("description") or "")[:MAX_DESC],
                "related": [{"uri": r["uri"], "label": r["label"]} for r in s["related"]][
                    :MAX_RELATED
                ],
            }
        )
    if not results:
        note = (
            f"no ESCO concept matches '{query}' (ESCO lists few vendor tools); "
            "try a broader generic term, e.g. the skill family"
        )
    else:
        note = f"{len(results)} ESCO concept(s), best first (ESCO, CC BY 4.0)"
    return {"results": results, "note": note}


async def check_cv_evidence(skill: str, cv_text: str) -> dict:
    """P(the CV shows `skill`) from the decision layer; None when it is disabled or down."""
    # Imported lazily: keeps this module importable without the decision layer's deps.
    from jobmatch.decisions.service import check_skill_evidence
    from jobmatch.schemas import SkillAssessment, SkillLevel

    skill = (skill or "").strip()
    out: dict[str, Any] = {"skill": skill, "p_evidence": None, "present": None}
    if not skill or not (cv_text or "").strip():
        return {**out, "note": "missing skill or CV text"}
    if not get_settings().decisions_enabled:
        return {**out, "note": "decision layer disabled: judge from the CV text yourself"}
    # match="partial" so that fusion v2 returns the decision model's probability as is.
    assessment = SkillAssessment(
        skill=skill, level=SkillLevel.required, match="partial", evidence=""
    )
    try:
        checks = await check_skill_evidence(cv_text, [assessment])
    except Exception as exc:  # noqa: BLE001 - a tool never raises
        log.debug("check_cv_evidence failed: %s", exc)
        checks = []
    if not checks:
        return {**out, "note": "decision layer unavailable: judge from the CV text yourself"}
    p = round(checks[0].p_evidence, 3)
    present = p >= 0.5
    verdict = "the CV shows it" if present else "no clear evidence in the CV"
    return {**out, "p_evidence": p, "present": present, "note": f"calibrated P={p}: {verdict}"}


async def search_learning_resources(skill: str, limit: int = 3) -> dict:
    """Lookup (skill name or alias, fuzzy) in the curated, link-checked local catalog."""
    q = normalize(skill or "")
    if not q:
        return {"results": [], "note": "empty skill"}
    index = _resource_index(data_dir() / RESOURCES_FILE)
    if not index:
        return {"results": [], "note": "learning-resource catalog unavailable"}
    # Score per canonical skill (best over all its entries' names/aliases), so that "Spark"
    # returns every "Apache Spark" entry; keep only the best-matching skill(s).
    per_skill: dict[str, float] = {}
    for e in index:
        skill_name = e.data["skill"]
        per_skill[skill_name] = max(per_skill.get(skill_name, 0.0), _best(q, e.labels))
    top = max(per_skill.values(), default=0.0)
    scored = [
        (per_skill[e.data["skill"]], e.data)
        for e in index
        if per_skill[e.data["skill"]] >= max(MIN_SCORE, top - 0.05)
    ]
    scored.sort(key=lambda x: (-x[0], not x[1].get("free"), KIND_ORDER.get(x[1]["kind"], 9)))
    keys = ("title", "url", "provider", "kind", "free", "level", "language", "est_hours")
    results = [{k: r.get(k) for k in keys} for _, r in scored[: _clip_limit(limit)]]
    if not results:
        note = (
            f"no curated resource for '{skill}': do not invent URLs; suggest the tool's "
            "official documentation in words, or search a broader skill"
        )
    else:
        note = f"{len(results)} of {len(scored)} curated resource(s), links checked"
    return {"results": results, "note": note}
