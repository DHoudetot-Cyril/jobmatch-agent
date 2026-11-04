"""HTTP response models (request body for decisions is `jobmatch.schemas.HumanDecision`)."""

from typing import Any

from pydantic import BaseModel, Field

from jobmatch.schemas import (
    AnalysisStatus,
    CandidateProfile,
    CoverLetter,
    JobOffer,
    MatchResult,
    Review,
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
    cover_letter: CoverLetter | None = None
    review: Review | None = None
    review_iterations: int = 0
    trace: list[str] = Field(default_factory=list, description="Visited graph nodes, in order")
    pending_review: dict[str, Any] | None = Field(
        default=None,
        description="Interrupt payload (cover letter + match) when awaiting human approval",
    )
    error: str | None = None


class HealthResponse(BaseModel):
    status: str = "ok"


class DependencyCheck(BaseModel):
    ok: bool
    detail: str | None = None


class ReadyResponse(BaseModel):
    ready: bool
    checks: dict[str, DependencyCheck]


class ErrorResponse(BaseModel):
    detail: str
