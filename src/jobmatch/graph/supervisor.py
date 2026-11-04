"""Top-level supervisor: decides which team runs next.

Routing is deterministic: `next_step` is a pure function of the state, no LLM call.
This is intentional. The pipeline is fixed (analysis -> matching -> career -> writing ->
human review), so asking an LLM "who goes next?" would add latency, tokens and a failure
mode (a wrong or unparsable routing decision) without adding any capability. A pure
function is reliable, cheap and trivially unit-testable, while each team keeps its agentic
behaviour internally (parallel parsing, writer/reviewer loop, and the autonomous tool-using
career advisor, whose ReAct loop is bounded in code inside `career_team`).

Routing table (first matching rule wins):

    status completed / rejected            -> END
    offer or profile missing               -> analysis_team
    match missing                          -> matching_team
    career_plan missing (advisor enabled)  -> career_team    (CAREER_ADVISOR_ENABLED=false
                                                              skips it: previous pipeline)
    cover_letter or review missing         -> writing_team   (also after human feedback,
                                                              which clears `review`)
    otherwise                              -> human_review
"""

from typing import Literal

from langgraph.graph import END
from langgraph.types import Command

from jobmatch.config import get_settings
from jobmatch.observability.metrics import instrument_node
from jobmatch.schemas import AnalysisStatus
from jobmatch.state import JobMatchState

Step = Literal[
    "analysis_team", "matching_team", "career_team", "writing_team", "human_review", "__end__"
]

FINAL_STATUSES = {AnalysisStatus.completed, AnalysisStatus.rejected}


def next_step(state: JobMatchState) -> Step:
    """Pure routing function of the state (and of the `career_advisor_enabled` setting)."""
    if state.get("status") in FINAL_STATUSES:
        return END
    if state.get("offer") is None or state.get("profile") is None:
        return "analysis_team"
    if state.get("match") is None:
        return "matching_team"
    if state.get("career_plan") is None and get_settings().career_advisor_enabled:
        return "career_team"
    if state.get("cover_letter") is None or state.get("review") is None:
        return "writing_team"
    return "human_review"


@instrument_node("supervisor")
def supervisor(state: JobMatchState) -> Command[Step]:
    goto = next_step(state)
    update: dict = {"trace": ["supervisor"]}
    if goto == "human_review":
        # Set before entering the node: `interrupt()` pauses inside it, so the status the
        # API reads while paused must already be in the state.
        update["status"] = AnalysisStatus.awaiting_approval
    elif goto != END:
        update["status"] = AnalysisStatus.running
    return Command(goto=goto, update=update)
