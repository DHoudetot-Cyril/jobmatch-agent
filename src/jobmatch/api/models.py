"""HTTP response models (request body for decisions is `jobmatch.schemas.HumanDecision`)."""

from typing import Any

from pydantic import BaseModel, Field, field_validator

from jobmatch.schemas import (
    AnalysisStatus,
    CandidateProfile,
    CareerPlan,
    ClaimCheck,
    CoverLetter,
    JobOffer,
    MatchResult,
    Review,
)


class ReviewView(Review):
    """API-facing `Review`: same data, with the decision-layer fields visible in OpenAPI.

    `schemas.Review` hides them (`SkipJsonSchema`) because its JSON schema doubles as the
    reviewer LLM's structured-output schema when the decision layer is disabled; the LLM-facing
    schema is left untouched.
    """

    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Exact quotes of letter claims still considered unsupported by the CV",
    )
    claim_checks: list[ClaimCheck] = Field(
        default_factory=list,
        description="Reviewer-flagged claims verified by the decision model (verify_claims)",
    )


class AnalysisAccepted(BaseModel):
    id: str = Field(description="Analysis id (= LangGraph thread id = Langfuse session id)")
    status: AnalysisStatus = AnalysisStatus.running


class AnalysisView(BaseModel):
    id: str
    status: AnalysisStatus
    offer: JobOffer | None = None
    profile: CandidateProfile | None = None
    match: MatchResult | None = None
    career_plan: CareerPlan | None = Field(
        default=None,
        description="Career advisor plan per skill gap (autonomous tool-using agent: ESCO "
        "taxonomy, verified transferable skills, catalog resources); null if disabled",
    )
    cover_letter: CoverLetter | None = None
    review: ReviewView | None = None
    review_iterations: int = 0
    trace: list[str] = Field(default_factory=list, description="Visited graph nodes, in order")
    pending_review: dict[str, Any] | None = Field(
        default=None,
        description="Interrupt payload (cover letter + match + career_plan) when awaiting "
        "human approval",
    )
    error: str | None = None

    @field_validator("review", mode="before")
    @classmethod
    def _review_to_view(cls, v: Any) -> Any:
        if isinstance(v, Review) and not isinstance(v, ReviewView):
            return v.model_dump()  # graph state holds a schemas.Review
        return v


class HealthResponse(BaseModel):
    status: str = "ok"


class DependencyCheck(BaseModel):
    ok: bool
    detail: str | None = None
    critical: bool = Field(
        default=True, description="false: the API stays ready when it fails (degraded mode)"
    )


class ReadyResponse(BaseModel):
    ready: bool = Field(description="All CRITICAL checks pass (database, llm)")
    degraded: bool = Field(
        default=False,
        description="A non-critical dependency is down (e.g. decisions: jevos unreachable -> "
        "analyses run LLM-only)",
    )
    checks: dict[str, DependencyCheck]


class ErrorResponse(BaseModel):
    detail: str
