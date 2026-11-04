"""LangGraph state. CONTRACT shared by graph (writer) and API (reader).

Graph public interface (implemented in `jobmatch.graph.build`):

    build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph

    - Input:  {"offer_text": str, "cv_text": str}   (documents are already converted to text
              by `jobmatch.documents.load_document` BEFORE invoking the graph)
    - Run:    await graph.ainvoke(input, config={"configurable": {"thread_id": analysis_id},
                                                 "callbacks": [langfuse_handler, ...]})
    - The graph pauses with `langgraph.types.interrupt(payload)` in node "human_review"
      where payload = {"cover_letter": CoverLetter.model_dump(), "match": MatchResult.model_dump()}.
    - Resume: await graph.ainvoke(Command(resume=HumanDecision(...).model_dump()), config=same)
      approved=True  -> status "completed"
      approved=False with feedback -> writer rewrites, reviewer, then human_review again
      approved=False without feedback -> status "rejected", graph ends
    - Node names (used as Prometheus `node` label & Langfuse span names):
      supervisor, parse_offer, parse_cv, score_match, analyze_gaps,
      write_letter, review_letter, human_review
"""

from typing import Annotated, TypedDict

from jobmatch.schemas import (
    AnalysisStatus,
    CandidateProfile,
    CoverLetter,
    JobOffer,
    MatchResult,
    Review,
)


def _append(left: list[str] | None, right: list[str] | None) -> list[str]:
    return (left or []) + (right or [])


class JobMatchState(TypedDict, total=False):
    # inputs
    offer_text: str
    cv_text: str
    # analysis team
    offer: JobOffer
    profile: CandidateProfile
    # matching team
    match: MatchResult
    # writing team
    cover_letter: CoverLetter
    review: Review
    review_iterations: int
    human_feedback: str | None
    # control
    status: AnalysisStatus
    error: str | None
    trace: Annotated[list[str], _append]  # ordered list of visited nodes (debug / API)
