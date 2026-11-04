"""Shared fixtures: a scripted fake chat model replacing `jobmatch.llm.get_llm`."""

from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from jobmatch.decisions.base import Decision, DecisionModel, Question
from jobmatch.schemas import (
    CandidateProfile,
    CoverLetter,
    Experience,
    JobOffer,
    Review,
    Skill,
    SkillGap,
    SkillLevel,
)

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"

INVALID = object()  # scripted response meaning "the model returned unparsable output"


class FakeLLM:
    """Minimal stand-in for the chat model, scripted per output schema.

    `responses[Schema]` is a list consumed in order (the last item repeats) whose items are a
    model instance, a callable `(messages) -> instance`, or `INVALID`. Plain `ainvoke` (used
    by the vision path) returns `text_response`. Every call is recorded in `calls`.
    """

    def __init__(self, responses: dict[type, list[Any]] | None = None, text_response: str = ""):
        self.responses = responses or {}
        self.text_response = text_response
        self.calls: dict[str, list[list[BaseMessage]]] = defaultdict(list)
        self.schemas: list[type[BaseModel]] = []  # exact schema of every structured call
        self.llm_kwargs: list[dict[str, Any]] = []  # get_llm() kwargs, one per model built

    def with_structured_output(self, schema: type[BaseModel], **kwargs: Any) -> "_Structured":
        assert kwargs.get("method") == "json_schema" and kwargs.get("include_raw") is True
        self.schemas.append(schema)
        return _Structured(self, schema)

    async def ainvoke(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> AIMessage:
        self.calls["text"].append(messages)
        return AIMessage(content=self.text_response)

    def count(self, schema: type) -> int:
        return len(self.calls[schema.__name__])


class _Structured:
    """Scripts are looked up by schema, then by its base classes (e.g. `ReviewerOutput` is
    answered by the `Review` script and recorded under "Review")."""

    def __init__(self, fake: FakeLLM, schema: type[BaseModel]):
        self.fake, self.schema = fake, schema
        self.key = next((c for c in schema.__mro__ if c in fake.responses), schema)

    async def ainvoke(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> dict:
        calls = self.fake.calls[self.key.__name__]
        calls.append(messages)
        script = self.fake.responses[self.key]
        item = script[min(len(calls) - 1, len(script) - 1)]
        if callable(item):
            item = item(messages)
        raw = AIMessage(
            content="{}",
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
        if item is INVALID:
            return {"raw": raw, "parsed": None, "parsing_error": ValueError("bad json")}
        parsed = self.schema.model_validate(item.model_dump())
        return {"raw": raw, "parsed": parsed, "parsing_error": None}


@pytest.fixture
def install_fake_llm(monkeypatch: pytest.MonkeyPatch) -> Callable[..., FakeLLM]:
    """Return a factory: `install_fake_llm(responses)` patches `get_llm` and returns the fake."""

    def install(responses: dict[type, list[Any]] | None = None, **kwargs: Any) -> FakeLLM:
        fake = FakeLLM(responses, **kwargs)

        def get_llm(temperature: float | None = None, **kw: Any) -> FakeLLM:
            fake.llm_kwargs.append({"temperature": temperature, **kw})
            return fake

        monkeypatch.setattr("jobmatch.llm.get_llm", get_llm)
        return fake

    return install


# --- fake decision model ("System 1") --------------------------------------------------------


class FakeDecisionModel(DecisionModel):
    """Answers yes/no questions with `p_yes(state, question_text) -> float`; records calls.
    With `error`, every call raises it (unreachable backend)."""

    name = "fake"

    def __init__(
        self,
        p_yes: Callable[[str, str], float] | None = None,
        error: Exception | None = None,
    ):
        self.p_yes = p_yes or (lambda state, q: 0.5)
        self.error = error
        self.questions: list[str] = []

    async def decide(self, state: str, question: Question) -> Decision:
        self.questions.append(question.question)
        if self.error is not None:
            raise self.error
        p = self.p_yes(state, question.question)
        return Decision(probs={"no": 1 - p, "yes": p}, backend=self.name)


@pytest.fixture(autouse=True)
def install_fake_decisions(monkeypatch: pytest.MonkeyPatch) -> Callable[..., FakeDecisionModel]:
    """Unit tests never reach a real decision backend.

    By default the backend is "unreachable" (graceful degradation path -> LLM-only results).
    `install_fake_decisions(p_yes)` installs a scripted one and returns it.
    """

    def install(p_yes=None, error: Exception | None = None) -> FakeDecisionModel:
        fake = FakeDecisionModel(p_yes, error)
        monkeypatch.setattr("jobmatch.decisions.service.get_decision_model", lambda: fake)
        return fake

    install(error=RuntimeError("no decision backend in unit tests"))
    return install


# --- canned domain objects -------------------------------------------------------------------


@pytest.fixture
def offer() -> JobOffer:
    return JobOffer(
        title="Data Engineer",
        company="Lumio Énergie",
        seniority="mid",
        skills=[
            Skill(name="Python", level=SkillLevel.required),
            Skill(name="Apache Spark", level=SkillLevel.required),
            Skill(name="Kafka", level=SkillLevel.nice_to_have),
        ],
        responsibilities=["Pipelines batch et streaming"],
        language="fr",
    )


@pytest.fixture
def profile() -> CandidateProfile:
    return CandidateProfile(
        name="Camille Martin",
        skills=["Python", "Apache Spark", "SQL"],
        experiences=[Experience(role="Data Engineer", company="Transfluvia", duration_months=32)],
    )


@pytest.fixture
def letter() -> CoverLetter:
    return CoverLetter(subject="Candidature Data Engineer", body="Madame, Monsieur, ...")


@pytest.fixture
def approved_review() -> Review:
    return Review(approved=True)


@pytest.fixture
def rejected_review() -> Review:
    return Review(approved=False, issues=["Claims Kafka experience, not in the CV"])


@pytest.fixture
def gap() -> SkillGap:
    return SkillGap(skill="Kafka", level=SkillLevel.nice_to_have, advice="Suivre un tutoriel")
