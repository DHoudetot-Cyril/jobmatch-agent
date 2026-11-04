"""Domain models shared by the graph, the API and the evals. CONTRACT: do not rename fields
without updating every consumer. Agents produce these via structured output.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema


class SkillLevel(StrEnum):
    required = "required"
    nice_to_have = "nice_to_have"


class Skill(BaseModel):
    name: str = Field(description="Canonical skill name, e.g. 'Python', 'Docker'")
    level: SkillLevel = SkillLevel.required


class JobOffer(BaseModel):
    """Produced by the offer parser (Analysis team)."""

    title: str
    company: str | None = None
    location: str | None = None
    seniority: str | None = Field(default=None, description="junior / mid / senior / lead")
    skills: list[Skill] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    language: str = Field(default="fr", description="ISO code of the offer's language")


class Experience(BaseModel):
    role: str
    company: str | None = None
    duration_months: int | None = None
    highlights: list[str] = Field(default_factory=list)


class CandidateProfile(BaseModel):
    """Produced by the CV parser (Analysis team)."""

    name: str | None = None
    headline: str | None = None
    skills: list[str] = Field(default_factory=list)
    experiences: list[Experience] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)


class SkillGap(BaseModel):
    skill: str
    level: SkillLevel
    advice: str = Field(description="Concrete way to close or mitigate the gap")


class SkillAssessment(BaseModel):
    """LLM rating of one offer skill (intermediate output of `score_match`)."""

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


class SkillCheck(BaseModel):
    """One offer skill verified by the decision model (node `verify_skills`).

    Fusion v2 (`decisions.service.fuse_skill`), llm_credit c in {1, 0.5, 0}, calibrated p:
      - c == 0.5 (LLM unsure)                  -> final = p    (resolution "resolved_partial")
      - c == 1 and p < decision_conflict_low   -> final = 0.5  (resolution "conflict")
      - c == 0 and p > decision_conflict_high  -> final = 0.5  (resolution "conflict")
      - otherwise                              -> final = c    (resolution "llm", LLM kept)
    The skill is a gap if final_credit < 0.5.
    """

    skill: str
    level: SkillLevel
    llm_credit: float = Field(description="LLM rating: full = 1, partial = 0.5, none = 0")
    p_evidence: float = Field(description="Calibrated P(the CV shows the skill), decision model")
    final_credit: float
    agree: bool = Field(description="LLM (credit >= 0.5) and decision model (p >= 0.5) agree")
    # Additive (v2); "llm" default so outputs written before v2 still load.
    resolution: Literal["llm", "resolved_partial", "conflict"] = Field(
        default="llm",
        description="llm: LLM rating kept; resolved_partial: 'partial' replaced by p_evidence; "
        "conflict: strong LLM / decision-model disagreement -> partial credit 0.5",
    )


class ClaimCheck(BaseModel):
    """One letter claim flagged by the reviewer, verified by the decision model."""

    claim: str
    p_supported: float = Field(description="Calibrated P(the CV supports the claim)")
    dismissed: bool = Field(description="True: the reviewer was wrong, the issue was removed")


class MatchResult(BaseModel):
    """Produced by the Matching team (scorer + gap analyst)."""

    score: int = Field(ge=0, le=100, description="Overall fit, 0-100")
    matched_skills: list[str] = Field(default_factory=list)
    gaps: list[SkillGap] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    summary: str = Field(description="2-3 sentence justification of the score")
    # Filled by `verify_skills` (empty when the decision layer is disabled / unavailable).
    skill_checks: list[SkillCheck] = Field(default_factory=list)


class Review(BaseModel):
    """Produced by the reviewer (Writing team)."""

    approved: bool
    issues: list[str] = Field(
        default_factory=list,
        description="Problems found, esp. any claim NOT supported by the CV (hallucination)",
    )
    suggestions: list[str] = Field(default_factory=list)
    # Decision layer (additive). Hidden from the LLM's JSON schema so that, with the decision
    # layer disabled, the reviewer sees exactly the historical schema.
    unsupported_claims: SkipJsonSchema[list[str]] = Field(
        default_factory=list,
        description="Exact quotes of letter claims still considered unsupported by the CV",
    )
    claim_checks: SkipJsonSchema[list[ClaimCheck]] = Field(default_factory=list)


class ReviewerOutput(Review):
    """LLM output schema of the reviewer when the decision layer is enabled.

    Unsupported claims are listed separately (exact quotes) so `verify_claims` can check each
    one against the CV; `issues` then holds only the OTHER problems.
    """

    issues: list[str] = Field(
        default_factory=list,
        description="Problems OTHER than unsupported claims (language, tailoring, tone...)",
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="EXACT quotes, copied verbatim from the letter, of every claim that is not "
        "supported by the candidate profile or the CV",
    )


class CoverLetter(BaseModel):
    """Produced by the writer (Writing team)."""

    subject: str
    body: str
    language: str = "fr"


class AnalysisStatus(StrEnum):
    running = "running"
    awaiting_approval = "awaiting_approval"  # paused on human-in-the-loop interrupt
    completed = "completed"
    rejected = "rejected"
    failed = "failed"


class HumanDecision(BaseModel):
    """Payload used to resume the graph after the human-in-the-loop interrupt."""

    approved: bool
    feedback: str | None = Field(
        default=None, description="If not approved: what to change; the writer rewrites"
    )
