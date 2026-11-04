"""Domain models shared by the graph, the API and the evals. CONTRACT: do not rename fields
without updating every consumer. Agents produce these via structured output.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


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


class MatchResult(BaseModel):
    """Produced by the Matching team (scorer + gap analyst)."""

    score: int = Field(ge=0, le=100, description="Overall fit, 0-100")
    matched_skills: list[str] = Field(default_factory=list)
    gaps: list[SkillGap] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    summary: str = Field(description="2-3 sentence justification of the score")


class Review(BaseModel):
    """Produced by the reviewer (Writing team)."""

    approved: bool
    issues: list[str] = Field(
        default_factory=list,
        description="Problems found, esp. any claim NOT supported by the CV (hallucination)",
    )
    suggestions: list[str] = Field(default_factory=list)


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
