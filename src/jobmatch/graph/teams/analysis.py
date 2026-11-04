"""Analysis team: turns raw texts into structured domain objects.

    START ─┬─> parse_offer ─┬─> END
           └─> parse_cv ────┘

Both parsers are independent, so they run in parallel (fan-out / fan-in): with the two
llama-server slots this roughly halves the team's latency.

The parsed offer's skills are de-duplicated (`dedupe_skills`): the LLM sometimes lists a skill
twice (e.g. "AWS" as required AND nice_to_have); every consumer then sees one entry per
case-insensitive name, with the strongest level (required wins).
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from jobmatch.graph.structured import invoke_structured
from jobmatch.graph.teams._common import TeamInput
from jobmatch.observability.metrics import instrument_node
from jobmatch.prompts.analysis import PARSE_CV_PROMPT, PARSE_OFFER_PROMPT
from jobmatch.schemas import CandidateProfile, JobOffer, Skill, SkillLevel
from jobmatch.state import JobMatchState


def dedupe_skills(skills: list[Skill]) -> list[Skill]:
    """One skill per case-insensitive name: first occurrence's name/position, strongest level."""
    merged: dict[str, Skill] = {}
    for skill in skills:
        key = skill.name.strip().casefold()
        if not key:
            continue
        if key not in merged:
            merged[key] = skill.model_copy()
        elif skill.level == SkillLevel.required:
            merged[key].level = SkillLevel.required
    return list(merged.values())


@instrument_node("parse_offer")
async def parse_offer(state: JobMatchState) -> dict:
    messages = PARSE_OFFER_PROMPT.format_messages(offer_text=state["offer_text"])
    offer = await invoke_structured("parse_offer", JobOffer, messages, temperature=0.0)
    offer.skills = dedupe_skills(offer.skills)
    return {"offer": offer, "trace": ["parse_offer"]}


@instrument_node("parse_cv")
async def parse_cv(state: JobMatchState) -> dict:
    messages = PARSE_CV_PROMPT.format_messages(cv_text=state["cv_text"])
    profile = await invoke_structured("parse_cv", CandidateProfile, messages, temperature=0.0)
    return {"profile": profile, "trace": ["parse_cv"]}


def build_analysis_team() -> CompiledStateGraph:
    g = StateGraph(JobMatchState, input_schema=TeamInput)
    g.add_node("parse_offer", parse_offer)
    g.add_node("parse_cv", parse_cv)
    g.add_edge(START, "parse_offer")
    g.add_edge(START, "parse_cv")
    g.add_edge("parse_offer", END)
    g.add_edge("parse_cv", END)
    return g.compile(name="analysis_team")
