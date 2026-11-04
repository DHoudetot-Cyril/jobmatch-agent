"""Writing team: cover-letter writer / reviewer loop.

    START ─> write_letter ─> review_letter ─┬─> END           (approved or budget exhausted)
                  ^                          │
                  └──────────────────────────┘ (issues found)

The reviewer is also the hallucination guard: it must flag every claim of the letter that is
not supported by the candidate profile / CV, and the writer must fix them. The loop is capped
by `settings.max_review_iterations` (number of reviews per writing round).

The writer also applies `human_feedback` (set by `human_review` when the human asks for a
rewrite); in that case it revises the previous draft instead of starting from scratch.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from jobmatch.config import get_settings
from jobmatch.graph.structured import invoke_structured
from jobmatch.graph.teams._common import TeamInput, to_json
from jobmatch.observability.metrics import instrument_node
from jobmatch.prompts.writing import (
    HUMAN_FEEDBACK_CHECK,
    REVIEW_LETTER_PROMPT,
    REVISION_CONTEXT,
    WRITE_LETTER_PROMPT,
)
from jobmatch.schemas import CoverLetter, Review
from jobmatch.state import JobMatchState


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {i}" for i in items) if items else "(none)"


def _revision_context(state: JobMatchState) -> str:
    """Previous draft + what to fix, or "" for a first draft."""
    letter = state.get("cover_letter")
    review = state.get("review")
    feedback = state.get("human_feedback")
    if letter is None or (review is None and not feedback):
        return ""
    return REVISION_CONTEXT.format(
        previous_letter=letter.body,
        issues=_bullets(review.issues if review else []),
        suggestions=_bullets(review.suggestions if review else []),
        human_feedback=feedback or "(none)",
    )


@instrument_node("write_letter")
async def write_letter(state: JobMatchState) -> dict:
    language = state["offer"].language
    messages = WRITE_LETTER_PROMPT.format_messages(
        language=language,
        offer_json=to_json(state["offer"]),
        profile_json=to_json(state["profile"]),
        match_json=to_json(state["match"]),
        revision_context=_revision_context(state),
    )
    letter = await invoke_structured("write_letter", CoverLetter, messages, temperature=0.5)
    letter.language = language
    return {"cover_letter": letter, "trace": ["write_letter"]}


@instrument_node("review_letter")
async def review_letter(state: JobMatchState) -> dict:
    letter = state["cover_letter"]
    feedback = state.get("human_feedback")
    messages = REVIEW_LETTER_PROMPT.format_messages(
        language=state["offer"].language,
        human_feedback_check=HUMAN_FEEDBACK_CHECK.format(feedback=feedback) if feedback else "",
        offer_json=to_json(state["offer"]),
        profile_json=to_json(state["profile"]),
        cv_text=state["cv_text"],
        subject=letter.subject,
        body=letter.body,
    )
    review = await invoke_structured("review_letter", Review, messages, temperature=0.0)
    if review.issues:  # an approval with open issues is inconsistent: issues win
        review.approved = False
    return {
        "review": review,
        "review_iterations": state.get("review_iterations", 0) + 1,
        "trace": ["review_letter"],
    }


def route_after_review(state: JobMatchState) -> str:
    """Stop when approved or when the review budget is exhausted."""
    if state["review"].approved:
        return END
    if state.get("review_iterations", 0) >= get_settings().max_review_iterations:
        return END
    return "write_letter"


def build_writing_team() -> CompiledStateGraph:
    g = StateGraph(JobMatchState, input_schema=TeamInput)
    g.add_node("write_letter", write_letter)
    g.add_node("review_letter", review_letter)
    g.add_edge(START, "write_letter")
    g.add_edge("write_letter", "review_letter")
    g.add_conditional_edges("review_letter", route_after_review, ["write_letter", END])
    return g.compile(name="writing_team")
