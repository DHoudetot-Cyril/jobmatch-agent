"""Build data/career/esco_skills.json: a small local snapshot of the ESCO skills taxonomy.

ONE-OFF build script (network). The career-advisor tools read the JSON offline at runtime.

ESCO (European Skills, Competences, Qualifications and Occupations) — (c) European Union,
https://esco.ec.europa.eu — licence CC BY 4.0. API: https://ec.europa.eu/esco/api

For every seed (skills appearing in data/samples and evals/dataset + common tech, data, devops,
support, accounting, logistics, healthcare skills) we search ESCO (`/search?type=skill`) in the
seed's language, keep the top results whose labels actually match the seed, then fetch each
concept ONCE (`/resource/skill`), keeping: uri, preferred_label {en, fr}, alt_labels {en, fr}
(trimmed), description (en, trimmed), related [{uri, label, relation}] from the real ESCO
skill-to-skill relations (broader/narrower, essential/optional).

Polite: concurrency <= 3, small sleep, retries with backoff; raw responses cached on disk
(default ~/.cache/jobmatch/esco, outside the repo; override with ESCO_CACHE_DIR).

    uv run python scripts/build_esco_snapshot.py            # uses the cache when present
    uv run python scripts/build_esco_snapshot.py --no-cache
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import os
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import httpx

API = "https://ec.europa.eu/esco/api"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "career" / "esco_skills.json"
CACHE = Path(os.environ.get("ESCO_CACHE_DIR", Path.home() / ".cache" / "jobmatch" / "esco"))
CONCURRENCY = 3
SLEEP_S = 0.15
SEARCH_LIMIT = 6  # results inspected per seed
KEEP_PER_SEED = 3  # matching results kept per seed

MAX_ALT = 8
MAX_DESC = 400
MAX_RELATED = 12
# ESCO skill-to-skill relations, in the order we prefer them for `related`.
RELATIONS = [
    ("broaderSkill", "broader"),
    ("narrowerSkill", "narrower"),
    ("hasEssentialSkill", "essential"),
    ("isEssentialForSkill", "essential_for"),
    ("hasOptionalSkill", "optional"),
    ("isOptionalForSkill", "optional_for"),
]

# fmt: off
SEEDS_EN = [
    # --- from data/samples + evals/dataset (offers & CVs) ---
    "Python", "SQL", "Apache Spark", "Apache Airflow", "Docker", "GitLab CI", "AWS", "Amazon S3",
    "AWS Glue", "Amazon Redshift", "Kafka", "dbt", "Terraform", "Django", "PostgreSQL",
    "REST API", "GitHub Actions", "pytest", "Kubernetes", "Celery", "Redis", "Linux", "Ansible",
    "Helm", "Prometheus", "Grafana", "Bash", "Zabbix", "VMware", "TCP/IP", "machine learning",
    "pandas", "scikit-learn", "statistics", "PyTorch", "TensorFlow", "MLflow",
    "time series forecasting", "React", "TypeScript", "JavaScript", "Next.js", "Jest",
    "Cypress", "GraphQL", "web accessibility", "code review", "mentoring", "Active Directory",
    "Windows", "Office 365", "ITIL", "GLPI", "help desk", "Sage", "Excel", "Pennylane",
    "general accounting", "VAT", "balance sheet", "payroll", "CACES", "forklift", "SAP",
    "warehouse management system", "inventory", "team management", "order picking",
    "occupational safety", "nursing", "patient care", "data warehouse", "data quality",
    "ETL", "CI/CD", "cloud computing",
    # --- common tech / data / devops ---
    "Java", "Scala", "C++", "C#", "Go programming", "Rust", "PHP", "Ruby", "R programming",
    "Kotlin", "Swift", "HTML", "CSS", "Angular", "Vue.js", "Node.js", "Spring framework",
    "Flask", "FastAPI", "Git", "Jenkins", "Microsoft Azure", "Google Cloud", "MySQL", "MongoDB",
    "Oracle database", "NoSQL", "Hadoop", "Apache Hive", "data mining", "data modelling",
    "data analysis", "data visualisation", "big data", "deep learning",
    "natural language processing", "computer vision", "artificial intelligence",
    "Power BI", "Tableau", "business intelligence", "data science", "database design",
    "software testing", "unit testing", "test automation", "Selenium", "agile project management",
    "Scrum", "Kanban", "DevOps", "infrastructure as code", "configuration management",
    "containerisation", "virtualisation", "network administration", "system administration",
    "cybersecurity", "firewall", "information security", "cloud security", "monitoring",
    "shell scripting", "Perl", "UML", "software architecture", "microservices", "API design",
    "web development", "frontend development", "user interface design", "UX design",
    "mobile development", "Android", "iOS", "object-oriented programming", "JSON", "XML",
    "LDAP", "DNS", "Microsoft Office", "spreadsheet software", "Word processing",
    "technical support", "ICT troubleshooting", "customer service", "incident management",
    "service level agreement", "remote support",
    # --- accounting / finance ---
    "accounting", "bookkeeping", "financial statements", "tax regulations", "auditing",
    "budgeting", "cost accounting", "accounts payable", "accounts receivable", "invoicing",
    "financial analysis", "ERP", "financial reporting",
    # --- logistics / supply chain ---
    "logistics", "supply chain management", "warehouse operations", "stock control",
    "shipping", "procurement", "lean manufacturing", "transport planning", "forklift operation",
    "health and safety regulations", "quality control", "KPI",
    # --- healthcare ---
    "medical terminology", "health informatics", "clinical data", "pharmacology",
    "first aid", "electronic health records", "public health", "epidemiology",
    "biostatistics",
    # --- soft / management ---
    "project management", "leadership", "communication", "teamwork", "problem solving",
    "time management", "negotiation", "presentation skills", "English language", "coaching",
    # --- ESCO phrasings of concepts that the tool names above miss (Kubernetes, Bash, ...) ---
    "cloud technologies", "use scripting programming", "handle helpdesk problems",
    "ICT help platforms", "web programming", "pick orders for dispatching",
    "manage a warehouse management system", "levels of software testing", "ICT testing",
    "ICT system programming", "operate forklift", "use spreadsheets software",
    "use microsoft office", "ICT incident management", "ICT security standards",
    "ICT network administration", "agile development", "ICT infrastructure",
    "ICT system integration", "statistical analysis", "utilise machine learning",
    "perform data analysis", "query languages", "online analytical processing",
    "health records management", "warehouse management", "tax legislation",
    "prepare financial statements", "manage staff", "lead a team", "supervise work",
    "software design patterns", "develop automated software tests", "use software libraries",
    "manage cloud data and storage", "deploy cloud resource", "automate cloud tasks",
    "design cloud architecture", "apply health and safety standards",
]
SEEDS_FR = [
    "comptabilité", "comptabilité générale", "gestion d'équipe", "encadrement", "paie",
    "logistique", "gestion des stocks", "préparation de commandes", "cariste",
    "chariot élévateur", "soins infirmiers", "tableur", "statistiques", "fiscalité",
    "apprentissage automatique", "tests logiciels", "sécurité au travail", "entrepôt",
    "support informatique", "administration système", "bilan comptable", "déclaration de TVA",
]
# Curated aliases: tool or French names that ESCO lacks or only reaches through homonyms,
# mapped BY HAND to the ESCO concept with exactly this English preferred label (an
# approximation, listed in data/career/README.md). Applied instead of the plain search.
CURATED = {
    "Excel": "use spreadsheets software", "Microsoft Excel": "use spreadsheets software",
    "tableur": "use spreadsheets software", "Office 365": "use microsoft office",
    "Bash": "use scripting programming", "shell scripting": "use scripting programming",
    "help desk": "handle helpdesk problems", "support informatique": "handle helpdesk problems",
    "technical support": "handle helpdesk problems",
    "system administration": "administer ICT system",
    "administration système": "administer ICT system",
    "CACES": "operate forklift", "cariste": "operate forklift", "forklift": "operate forklift",
    "order picking": "pick orders for dispatching",
    "préparation de commandes": "pick orders for dispatching",
    "WMS": "manage a warehouse management system",
    "warehouse management system": "manage a warehouse management system",
    "AWS": "cloud technologies", "Microsoft Azure": "cloud technologies",
    "Google Cloud": "cloud technologies", "cloud computing": "cloud technologies",
    "Git": "tools for software configuration management",
    "team management": "lead a team", "gestion d'équipe": "lead a team",
    "encadrement": "lead a team", "management": "lead a team",
    "comptabilité générale": "accounting", "general accounting": "accounting",
    "Power BI": "business intelligence", "Tableau": "business intelligence",
    "Linux": "operating systems",
}
# fmt: on


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.casefold())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9+#./ ]+", " ", s).strip()


def tokens(s: str) -> set[str]:
    return {t for t in re.split(r"[ ./]+", norm(s)) if len(t) > 1 or t in {"c", "r"}}


def matches(seed: str, res: dict) -> bool:
    """A search hit is kept only if one of its labels really relates to the seed.

    One-word seeds must match a PREFERRED label: alternative labels are full of homonyms
    ("Excel" -> "demand performers excel").
    """
    st = tokens(seed)
    pref = [res.get("title", ""), *(res.get("preferredLabel") or {}).values()]
    labels = [x for x in pref if x] if len(st) == 1 else labels_of(res)
    for lab in labels:
        lt = tokens(lab)
        if st and st <= lt:
            return True
        if SequenceMatcher(None, norm(seed), norm(lab)).ratio() >= 0.8:
            return True
    return False


def trim(s: str, n: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1].rsplit(" ", 1)[0] + "…"


class Client:
    def __init__(self, use_cache: bool) -> None:
        self.use_cache = use_cache
        self.http = httpx.AsyncClient(
            timeout=30, headers={"User-Agent": "jobmatch-agent ESCO snapshot builder (one-off)"}
        )
        self.sem = asyncio.Semaphore(CONCURRENCY)
        CACHE.mkdir(parents=True, exist_ok=True)

    async def get(self, path: str, params: dict) -> dict:
        key = hashlib.sha1(json.dumps([path, params], sort_keys=True).encode()).hexdigest()
        f = CACHE / f"{key}.json"
        if self.use_cache and f.exists():
            return json.loads(f.read_text())
        async with self.sem:
            for attempt in range(4):
                try:
                    r = await self.http.get(f"{API}{path}", params=params)
                    if r.status_code == 200:
                        data = r.json()
                        f.write_text(json.dumps(data))
                        await asyncio.sleep(SLEEP_S)
                        return data
                    if r.status_code == 404:
                        return {}
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(1.5 * 2**attempt)
        raise RuntimeError(f"ESCO request failed: {path} {params}")


def labels_of(res: dict) -> list[str]:
    out = [res.get("title", "")]
    out += list((res.get("preferredLabel") or {}).values())
    for lang in ("en", "fr"):
        out += (res.get("alternativeLabel") or {}).get(lang, [])
    return [x for x in out if x]


def concept(res: dict) -> dict:
    pref = res.get("preferredLabel") or {}
    alt = res.get("alternativeLabel") or {}
    desc = ((res.get("description") or {}).get("en") or {}).get("literal", "")
    related, seen = [], {res["uri"]}
    links = res.get("_links") or {}
    for key, rel in RELATIONS:
        for link in links.get(key) or []:
            if link["uri"] not in seen and len(related) < MAX_RELATED:
                seen.add(link["uri"])
                related.append({"uri": link["uri"], "label": link["title"], "relation": rel})
    group = (links.get("broaderHierarchyConcept") or [{}])[0].get("title")
    return {
        "uri": res["uri"],
        "preferred_label": {"en": pref.get("en", res.get("title", "")), "fr": pref.get("fr")},
        "alt_labels": {
            lang: [trim(a, 80) for a in alt.get(lang, [])][:MAX_ALT] for lang in ("en", "fr")
        },
        "description": trim(desc, MAX_DESC),
        "skill_group": group,
        "related": related,
    }


async def resolve_curated(c: Client, seed: str, target: str) -> tuple[str, list[str]]:
    data = await c.get(
        "/search", {"text": target, "type": "skill", "language": "en", "limit": SEARCH_LIMIT}
    )
    for h in (data.get("_embedded") or {}).get("results") or []:
        if norm(h["title"]) == norm(target):
            return seed, [h["uri"]]
    return seed, []


async def resolve_seed(c: Client, seed: str, lang: str) -> tuple[str, list[str]]:
    if seed in CURATED:
        return await resolve_curated(c, seed, CURATED[seed])
    data = await c.get(
        "/search", {"text": seed, "type": "skill", "language": lang, "limit": SEARCH_LIMIT}
    )
    hits = (data.get("_embedded") or {}).get("results") or []
    uris = []
    for h in hits:
        res = await c.get("/resource/skill", {"uri": h["uri"], "language": "en"})
        if res and matches(seed, res):
            uris.append(h["uri"])
        if len(uris) >= KEEP_PER_SEED:
            break
    return seed, uris


async def main(use_cache: bool) -> None:
    c = Client(use_cache)
    seeds = [(s, "en") for s in SEEDS_EN] + [(s, "fr") for s in SEEDS_FR]
    seeds += [(s, "en") for s in CURATED if s not in SEEDS_EN and s not in SEEDS_FR]
    results = await asyncio.gather(*(resolve_seed(c, s, lang) for s, lang in seeds))
    concepts: dict[str, dict] = {}
    seed_map: dict[str, list[str]] = {}
    for seed, uris in results:
        seed_map[seed] = uris
        for u in uris:
            if u not in concepts:
                res = await c.get("/resource/skill", {"uri": u, "language": "en"})
                concepts[u] = concept(res)
    await c.http.aclose()
    not_found = sorted(s for s, u in seed_map.items() if not u)
    out = {
        "source": "ESCO (European Skills, Competences, Qualifications and Occupations), "
        "https://esco.ec.europa.eu — (c) European Union, licence CC BY 4.0. "
        "Subset fetched via the ESCO API; labels/descriptions trimmed.",
        "license": "CC-BY-4.0",
        "built_at": dt.date.today().isoformat(),
        "stats": {
            "seeds": len(seeds),
            "found": len(seeds) - len(not_found),
            "concepts": len(concepts),
        },
        "seeds_not_found": not_found,
        "curated_aliases": {s: CURATED[s] for s in sorted(CURATED) if seed_map.get(s)},
        "seeds": seed_map,
        "skills": sorted(concepts.values(), key=lambda x: x["preferred_label"]["en"].casefold()),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps(out["stats"]), f"{OUT.stat().st_size / 1e6:.2f} MB")
    print("not found:", ", ".join(not_found))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-cache", action="store_true")
    asyncio.run(main(use_cache=not ap.parse_args().no_cache))
