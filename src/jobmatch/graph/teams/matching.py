"""Matching team: grounded match score, verified by the decision model, then skill gaps.

    START ─> score_match ─> verify_skills ─> analyze_gaps ─> END

Grounding: the LLM does not pick the score directly. It assesses every offer skill
(full / partial / none, with CV evidence) plus an experience-fit rating, and the score is
computed in code from a fixed rubric (`compute_score`): required skills weigh 70%,
nice-to-have 15%, experience fit 15%. The score is thus explainable and reproducible.

Verification ("System 2" LLM + "System 1" decision model): `verify_skills` asks the decision
model, for every offer skill, "does the CV show evidence of <skill>?" (calibrated probability
p) and fuses it with the LLM rating c in {1, 0.5, 0} (fusion v2, `decisions.service.fuse_skill`):
  - c == 0.5 (LLM unsure / partial)            -> final = p   (the decision model resolves it)
  - c == 1 and p < DECISION_CONFLICT_LOW (0.2)  -> final = 0.5 (strong disagreement, flagged)
  - c == 0 and p > DECISION_CONFLICT_HIGH (0.8) -> final = 0.5 (flagged)
  - otherwise                                   -> final = c   (LLM kept, no systematic bias)
Each `SkillCheck.resolution` records which case applied. The score is recomputed with the same
rubric, a skill is matched if final_credit >= 0.5 and a gap otherwise. If the decision layer is
disabled or unavailable, the LLM-only result is kept as is.

Offer skills are unique per case-insensitive name (`analysis.dedupe_skills`); the level map
below still lets "required" win in case an older checkpoint holds duplicates.
"""

from collections.abc import Iterable

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from jobmatch.config import get_settings
from jobmatch.decisions import service as decisions
from jobmatch.graph.structured import invoke_structured
from jobmatch.graph.teams._common import TeamInput, to_json
from jobmatch.observability.metrics import instrument_node
from jobmatch.prompts.matching import ANALYZE_GAPS_PROMPT, SCORE_MATCH_PROMPT
from jobmatch.schemas import (  # noqa: F401 - SkillAssessment/ScoreAssessment re-exported
    MatchResult,
    ScoreAssessment,
    SkillAssessment,
    SkillGap,
    SkillLevel,
)
from jobmatch.state import JobMatchState

WEIGHT_REQUIRED = 0.70
WEIGHT_NICE_TO_HAVE = 0.15
WEIGHT_EXPERIENCE = 0.15
MATCH_VALUE = decisions.LLM_CREDIT  # full = 1, partial = 0.5, none = 0
GAP_THRESHOLD = 0.5  # verified skills: gap if final_credit < 0.5


class GapAnalysis(BaseModel):
    gaps: list[SkillGap] = Field(default_factory=list)


def weighted_score(credits: Iterable[tuple[SkillLevel, float]], experience_fit: int) -> int:
    """Weighted rubric over (level, credit in [0, 1]) pairs; weights of an empty skill
    category are redistributed."""
    credits = list(credits)
    parts: list[tuple[float, float]] = []  # (weight, coverage in [0, 1])
    for level, weight in (
        (SkillLevel.required, WEIGHT_REQUIRED),
        (SkillLevel.nice_to_have, WEIGHT_NICE_TO_HAVE),
    ):
        values = [c for lvl, c in credits if lvl == level]
        if values:
            parts.append((weight, sum(values) / len(values)))
    parts.append((WEIGHT_EXPERIENCE, experience_fit / 100))
    total_weight = sum(w for w, _ in parts)
    return round(100 * sum(w * c for w, c in parts) / total_weight)


def compute_score(assessment: ScoreAssessment) -> int:
    """LLM-only score (rubric applied to the full / partial / none ratings)."""
    return weighted_score(
        ((s.level, MATCH_VALUE[s.match]) for s in assessment.skills), assessment.experience_fit
    )


def _merge_advice(gaps: list[SkillGap], advised: list[SkillGap]) -> list[SkillGap]:
    """Keep the skill names/levels computed by `score_match`, take only the LLM's advice.

    The LLM may decorate names (e.g. "Kafka (nice_to_have)") or reorder entries, so advice
    is matched by name prefix, then by position; the original evidence is the fallback.
    """
    merged = []
    for i, gap in enumerate(gaps):
        key = gap.skill.casefold()
        found = next((a for a in advised if a.skill.casefold().startswith(key)), None)
        if found is None and len(advised) == len(gaps):
            found = advised[i]
        merged.append(gap.model_copy(update={"advice": found.advice if found else gap.advice}))
    return merged


@instrument_node("score_match")
async def score_match(state: JobMatchState) -> dict:
    offer = state["offer"]
    messages = SCORE_MATCH_PROMPT.format_messages(
        language=offer.language,
        offer_json=to_json(offer),
        profile_json=to_json(state["profile"]),
    )
    assessment = await invoke_structured("score_match", ScoreAssessment, messages, temperature=0)
    # Trust the offer for the skill level rather than the LLM's copy of it (required wins).
    levels: dict[str, SkillLevel] = {}
    for s in offer.skills:
        key = s.name.strip().casefold()
        if levels.get(key) != SkillLevel.required:
            levels[key] = s.level
    unique: dict[str, SkillAssessment] = {}  # one rating per skill (the LLM may repeat one)
    for s in assessment.skills:
        key = s.skill.strip().casefold()
        s.level = levels.get(key, s.level)
        unique.setdefault(key, s)
    assessment.skills = list(unique.values())
    match = MatchResult(
        score=compute_score(assessment),
        matched_skills=[s.skill for s in assessment.skills if s.match != "none"],
        # Provisional gaps (no advice yet); analyze_gaps replaces them.
        gaps=[
            SkillGap(skill=s.skill, level=s.level, advice=s.evidence)
            for s in assessment.skills
            if s.match != "full"
        ],
        strengths=assessment.strengths,
        summary=assessment.summary,
    )
    return {"match": match, "score_assessment": assessment, "trace": ["score_match"]}


@instrument_node("verify_skills")
async def verify_skills(state: JobMatchState) -> dict:
    """Fuse the LLM skill ratings with the decision model's calibrated evidence check."""
    assessment = state.get("score_assessment")
    if not get_settings().decisions_enabled or assessment is None or not assessment.skills:
        return {"trace": ["verify_skills"]}
    checks = await decisions.check_skill_evidence(state["cv_text"], assessment.skills)
    if not checks:  # decision layer unavailable: keep the LLM-only result untouched
        return {"trace": ["verify_skills"]}
    match = state["match"]
    gap_names = {c.skill for c in checks if c.final_credit < GAP_THRESHOLD}
    match = match.model_copy(
        update={
            "score": weighted_score(
                ((c.level, c.final_credit) for c in checks), assessment.experience_fit
            ),
            "matched_skills": [c.skill for c in checks if c.final_credit >= GAP_THRESHOLD],
            # gaps ⊆ provisional gaps (a "full" skill has final_credit >= 0.5), so the
            # evidence collected by score_match is kept as provisional advice.
            "gaps": [g for g in match.gaps if g.skill in gap_names],
            "skill_checks": checks,
        }
    )
    return {"match": match, "trace": ["verify_skills"]}


@instrument_node("analyze_gaps")
async def analyze_gaps(state: JobMatchState) -> dict:
    match = state["match"]
    if not match.gaps:
        return {"trace": ["analyze_gaps"]}
    offer = state["offer"]
    gaps = "\n".join(f"- {g.skill} ({g.level.value}): {g.advice}" for g in match.gaps)
    messages = ANALYZE_GAPS_PROMPT.format_messages(
        language=offer.language,
        offer_json=to_json(offer),
        profile_json=to_json(state["profile"]),
        gaps=gaps,
    )
    analysis = await invoke_structured("analyze_gaps", GapAnalysis, messages)
    match = match.model_copy(update={"gaps": _merge_advice(match.gaps, analysis.gaps)})
    return {"match": match, "trace": ["analyze_gaps"]}


def build_matching_team() -> CompiledStateGraph:
    g = StateGraph(JobMatchState, input_schema=TeamInput)
    g.add_node("score_match", score_match)
    g.add_node("verify_skills", verify_skills)
    g.add_node("analyze_gaps", analyze_gaps)
    g.add_edge(START, "score_match")
    g.add_edge("score_match", "verify_skills")
    g.add_edge("verify_skills", "analyze_gaps")
    g.add_edge("analyze_gaps", END)
    return g.compile(name="matching_team")
