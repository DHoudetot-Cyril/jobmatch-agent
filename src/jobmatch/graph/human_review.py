"""Human-in-the-loop approval of the cover letter (node `human_review`).

Pauses the graph with `interrupt(payload)`; the API resumes it with
`Command(resume=HumanDecision(...).model_dump())` (see `jobmatch.state`).
"""

from langgraph.types import interrupt

from jobmatch.observability.metrics import instrument_node
from jobmatch.schemas import AnalysisStatus, HumanDecision
from jobmatch.state import JobMatchState


@instrument_node("human_review")
def human_review(state: JobMatchState) -> dict:
    payload = {
        "cover_letter": state["cover_letter"].model_dump(),
        "match": state["match"].model_dump(),
    }
    # On resume the node re-runs from the top and `interrupt` returns the resume value.
    decision = HumanDecision.model_validate(interrupt(payload))
    update: dict = {"trace": ["human_review"]}
    if decision.approved:
        update["status"] = AnalysisStatus.completed
    elif decision.feedback and decision.feedback.strip():
        # Rewrite round: clearing `review` sends the supervisor back to the writing team,
        # with a fresh review budget.
        update |= {
            "status": AnalysisStatus.running,
            "human_feedback": decision.feedback.strip(),
            "review": None,
            "review_iterations": 0,
        }
    else:
        update["status"] = AnalysisStatus.rejected
    return update
