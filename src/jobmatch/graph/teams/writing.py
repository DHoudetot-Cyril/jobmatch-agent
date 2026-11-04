"""Writing team: cover-letter writer / reviewer loop.

    START ─> write_letter ─> review_letter ─> verify_claims ─┬─> END  (approved / budget out)
                  ^                                          │
                  └──────────────────────────────────────────┘ (issues remain)

The reviewer is also the hallucination guard: it must flag every claim of the letter that is
not supported by the candidate profile / CV, and the writer must fix them. The loop is capped
by `settings.max_review_iterations` (number of reviews per writing round).

`verify_claims` targets the opposite failure, an over-strict reviewer flagging correct facts
(observed in the evals: it never approved). With the decision layer enabled, the reviewer
lists unsupported claims as exact quotes; each one is checked against the CV by the decision
model and, if calibrated P(supported) >= `claim_dismiss_threshold` (0.8), dismissed as a
reviewer false positive. The review is approved when no issue remains. With the decision
layer disabled the reviewer uses its original prompt/schema and `verify_claims` is a no-op.

The writer also applies `human_feedback` (set by `human_review` when the human asks for a
rewrite); in that case it revises the previous draft instead of starting from scratch.
"""

import logging
import re

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from jobmatch.config import get_settings
from jobmatch.decisions import service as decisions
from jobmatch.graph.structured import invoke_structured
from jobmatch.graph.teams._common import TeamInput, to_json
from jobmatch.observability.metrics import CLAIMS_DISMISSED, instrument_node
from jobmatch.prompts.writing import (
    HUMAN_FEEDBACK_CHECK,
    REVIEW_LETTER_CLAIMS_PROMPT,
    REVIEW_LETTER_PROMPT,
    REVISION_CONTEXT,
    WRITE_LETTER_PROMPT,
)
from jobmatch.schemas import CoverLetter, Review, ReviewerOutput
from jobmatch.state import JobMatchState

log = logging.getLogger(__name__)


def claim_issue(claim: str) -> str:
    """The issue text generated for an unsupported claim (removed again if dismissed)."""
    return f'Unsupported claim (not backed by the CV): "{claim}"'


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[\"'«»“”‘’]", "", text)).strip().casefold()


def _merge_claims(out: Review) -> Review:
    """Reviewer output (claims apart) -> Review whose issues include one per claim.

    The LLM sometimes repeats a claim in `issues` despite the prompt: an issue that contains
    a flagged quote is dropped, the generated `claim_issue` represents it.
    """
    claims = [c.strip() for c in out.unsupported_claims if c.strip()]
    keys = [_norm(c) for c in claims]
    other = [i for i in out.issues if not any(k and k in _norm(i) for k in keys)]
    issues = [*other, *(claim_issue(c) for c in claims)]
    return Review(
        approved=out.approved and not issues,
        issues=issues,
        suggestions=out.suggestions,
        unsupported_claims=claims,
    )


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
    prompt_vars = {
        "language": state["offer"].language,
        "human_feedback_check": HUMAN_FEEDBACK_CHECK.format(feedback=feedback) if feedback else "",
        "offer_json": to_json(state["offer"]),
        "profile_json": to_json(state["profile"]),
        "cv_text": state["cv_text"],
        "subject": letter.subject,
        "body": letter.body,
    }
    if get_settings().decisions_enabled:
        out = await invoke_structured(
            "review_letter",
            ReviewerOutput,
            REVIEW_LETTER_CLAIMS_PROMPT.format_messages(**prompt_vars),
            temperature=0.0,
        )
        review = _merge_claims(out)
    else:  # historical behaviour, unchanged
        review = await invoke_structured(
            "review_letter",
            Review,
            REVIEW_LETTER_PROMPT.format_messages(**prompt_vars),
            temperature=0.0,
        )
    if review.issues:  # an approval with open issues is inconsistent: issues win
        review.approved = False
    return {
        "review": review,
        "review_iterations": state.get("review_iterations", 0) + 1,
        "trace": ["review_letter"],
    }


@instrument_node("verify_claims")
async def verify_claims(state: JobMatchState) -> dict:
    """Dismiss the reviewer's unsupported-claim issues that the decision model finds supported."""
    review = state["review"]
    if not get_settings().decisions_enabled or not review.unsupported_claims:
        return {"trace": ["verify_claims"]}
    checks = await decisions.check_claims(state["cv_text"], review.unsupported_claims)
    if not checks:  # decision layer unavailable: keep the reviewer's verdict
        return {"trace": ["verify_claims"]}
    dismissed = {c.claim for c in checks if c.dismissed}
    for c in checks:
        if c.dismissed:
            log.info("verify_claims: dismissed (p_supported=%.2f): %s", c.p_supported, c.claim)
    CLAIMS_DISMISSED.inc(len(dismissed))
    removed = {claim_issue(c) for c in dismissed}
    issues = [i for i in review.issues if i not in removed]
    review = review.model_copy(
        update={
            "issues": issues,
            "unsupported_claims": [c for c in review.unsupported_claims if c not in dismissed],
            "claim_checks": checks,
            "approved": not issues,
        }
    )
    return {"review": review, "trace": ["verify_claims"]}


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
    g.add_node("verify_claims", verify_claims)
    g.add_edge(START, "write_letter")
    g.add_edge("write_letter", "review_letter")
    g.add_edge("review_letter", "verify_claims")
    g.add_conditional_edges("verify_claims", route_after_review, ["write_letter", END])
    return g.compile(name="writing_team")
