"""Matching team: grounded match score, then actionable skill gaps.

    START ─> score_match ─> analyze_gaps ─> END

Grounding: the LLM does not pick the score directly. It assesses every offer skill
(full / partial / none, with CV evidence) plus an experience-fit rating, and the score is
computed in code from a fixed rubric (`compute_score`): required skills weigh 70%,
nice-to-have 15%, experience fit 15%. The score is thus explainable and reproducible.
"""

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from jobmatch.graph.structured import invoke_structured
from jobmatch.graph.teams._common import TeamInput, to_json
from jobmatch.observability.metrics import instrument_node
from jobmatch.prompts.matching import ANALYZE_GAPS_PROMPT, SCORE_MATCH_PROMPT
from jobmatch.schemas import MatchResult, SkillGap, SkillLevel
from jobmatch.state import JobMatchState

WEIGHT_REQUIRED = 0.70
WEIGHT_NICE_TO_HAVE = 0.15
WEIGHT_EXPERIENCE = 0.15
MATCH_VALUE = {"full": 1.0, "partial": 0.5, "none": 0.0}


class SkillAssessment(BaseModel):
    skill: str
    level: SkillLevel
    match: Literal["full", "partial", "none"]
    evidence: str = Field(description="CV passage supporting the decision, or what is missing")


class ScoreAssessment(BaseModel):
    """Intermediate output of `score_match` (the score itself is computed in code)."""

    skills: list[SkillAssessment] = Field(default_factory=list)
    experience_fit: int = Field(ge=0, le=100)
    strengths: list[str] = Field(default_factory=list)
    summary: str


class GapAnalysis(BaseModel):
    gaps: list[SkillGap] = Field(default_factory=list)


def compute_score(assessment: ScoreAssessment) -> int:
    """Weighted rubric; weights of an empty skill category are redistributed."""
    parts: list[tuple[float, float]] = []  # (weight, coverage in [0, 1])
    for level, weight in (
        (SkillLevel.required, WEIGHT_REQUIRED),
        (SkillLevel.nice_to_have, WEIGHT_NICE_TO_HAVE),
    ):
        values = [MATCH_VALUE[s.match] for s in assessment.skills if s.level == level]
        if values:
            parts.append((weight, sum(values) / len(values)))
    parts.append((WEIGHT_EXPERIENCE, assessment.experience_fit / 100))
    total_weight = sum(w for w, _ in parts)
    return round(100 * sum(w * c for w, c in parts) / total_weight)


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
    # Trust the offer for the skill level rather than the LLM's copy of it.
    levels = {s.name.casefold(): s.level for s in offer.skills}
    for s in assessment.skills:
        s.level = levels.get(s.skill.casefold(), s.level)
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
    return {"match": match, "trace": ["score_match"]}


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
    g.add_node("analyze_gaps", analyze_gaps)
    g.add_edge(START, "score_match")
    g.add_edge("score_match", "analyze_gaps")
    g.add_edge("analyze_gaps", END)
    return g.compile(name="matching_team")
