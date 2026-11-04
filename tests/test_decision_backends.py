"""Laya / jevos decision backends: mapping logic with the model mocked, real runs marked `llm`."""

import json
import math
from typing import Any

import httpx
import pytest

from jobmatch.decisions.base import Question
from jobmatch.decisions.jevos_backend import (
    DEFAULT_BINARY,
    JevosDecisionModel,
    from_jev_answer,
    to_jev_question,
)
from jobmatch.decisions.laya_backend import (
    LayaDecisionModel,
    from_laya_answer,
    normalize,
    to_laya_question,
)

YES_NO = Question(kind="yes_no", question="Does this CV show evidence of Docker?")
CHOICE = Question(
    kind="choice",
    question="What is the candidate's seniority?",
    options={"junior": "0-2 years", "mid": "3-5 years", "senior": "6+ years"},
)
SCORE = Question(
    kind="score",
    question="How strong is the DevOps experience?",
    options={"none": "", "basic": "some scripts", "solid": "CI/CD in prod", "expert": ""},
)
CV = "Jane Doe, backend engineer, 6 years. Containerised services with Docker on Kubernetes."


def _sums_to_one(probs: dict[str, float]) -> bool:
    return math.isclose(sum(probs.values()), 1.0, abs_tol=1e-9)


# --- shared helpers ---------------------------------------------------------------------------


def test_normalize_renormalises_and_handles_degenerate() -> None:
    assert normalize({"a": 0.2, "b": 0.2}, ["a", "b"]) == {"a": 0.5, "b": 0.5}
    assert normalize({}, ["a", "b", "c"]) == pytest.approx({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})
    assert normalize({"a": -1.0, "b": 1.0}, ["a", "b"]) == {"a": 0.0, "b": 1.0}


# --- Laya mapping -----------------------------------------------------------------------------


def test_to_laya_question() -> None:
    assert to_laya_question(YES_NO) == {"type": "noul", "instructions": YES_NO.question}
    choice = to_laya_question(CHOICE)
    assert choice["type"] == "choice"
    assert choice["criteria"] == CHOICE.options
    score = to_laya_question(SCORE)
    assert score["type"] == "score"
    assert score["criteria"] == ["none", "basic: some scripts", "solid: CI/CD in prod", "expert"]


def test_from_laya_answer() -> None:
    yn = from_laya_answer(YES_NO, {"type": "noul", "noul": 0.88})
    assert yn == pytest.approx({"no": 0.12, "yes": 0.88})
    assert list(yn) == ["no", "yes"]
    ch = from_laya_answer(
        CHOICE, {"probabilities": {"junior": 0.1, "mid": 0.2, "senior": 0.7001}}
    )  # rounded output -> renormalised
    assert list(ch) == CHOICE.option_labels and _sums_to_one(ch)
    sc = from_laya_answer(SCORE, {"probabilities": {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4}})
    assert sc == pytest.approx({"none": 0.1, "basic": 0.2, "solid": 0.3, "expert": 0.4})


class FakeLayaAgent:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def system_one(self, state: str, questions: dict[str, Any], **kw: Any) -> dict[str, Any]:
        self.calls.append({"state": state, "questions": questions, **kw})
        q = questions["q"]
        if q["type"] == "noul":
            answer: dict[str, Any] = {"type": "noul", "noul": 0.9}
        elif q["type"] == "choice":
            keys = list(q["criteria"])
            answer = {"type": "choice", "probabilities": {k: 1 / len(keys) for k in keys}}
        else:
            n = len(q["criteria"])
            answer = {"type": "score", "probabilities": {str(i): 1 / n for i in range(n)}}
        return {"answers": {"q": answer}, "usage": {"truncated": False}}


@pytest.mark.parametrize("question", [YES_NO, CHOICE, SCORE])
async def test_laya_decide_with_mocked_agent(question: Question) -> None:
    model = LayaDecisionModel(max_len=2048)
    fake = FakeLayaAgent()
    model._agent = fake
    d = await model.decide(CV, question)
    assert d.backend == "laya"
    assert list(d.probs) == question.option_labels and _sums_to_one(d.probs)
    assert fake.calls[0]["max_len"] == 2048
    assert fake.calls[0]["state"] == CV


def test_laya_checkpoint_selection() -> None:
    assert LayaDecisionModel().repo == "convaiinnovations/laya-multilingual"
    assert LayaDecisionModel("english").repo == "convaiinnovations/laya"
    typed = LayaDecisionModel("typed-decisions")
    assert typed.repo == "convaiinnovations/laya-typed-decisions" and typed.revision
    assert LayaDecisionModel("./my-ckpt").repo == "./my-ckpt"
    with pytest.raises(ValueError):
        LayaDecisionModel(max_len=10_000)


# --- jevos mapping ----------------------------------------------------------------------------


def test_to_jev_question() -> None:
    assert to_jev_question(YES_NO) == {"type": "noul", "instructions": YES_NO.question}
    assert to_jev_question(CHOICE)["criteria"] == CHOICE.options
    assert to_jev_question(SCORE)["criteria"][1] == "basic: some scripts"
    too_many = Question(kind="score", question="?", options={str(i): f"l{i}" for i in range(11)})
    with pytest.raises(ValueError):
        to_jev_question(too_many)


def test_from_jev_answer() -> None:
    assert from_jev_answer(YES_NO, {"noul": 0.97}) == pytest.approx({"no": 0.03, "yes": 0.97})
    sc = from_jev_answer(SCORE, {"probabilities": {"0": 0.5, "1": 0.5, "2": 0.0, "3": 0.0}})
    assert sc == {"none": 0.5, "basic": 0.5, "solid": 0.0, "expert": 0.0}


async def test_jevos_decide_with_mocked_server() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ready", "model": "jevos-v5"})
        body = json.loads(request.content)
        seen.append(body)
        probs = {k: 1 / 3 for k in body["questions"]["q"]["criteria"]}
        return httpx.Response(
            200, json={"answers": {"q": {"type": "choice", "probabilities": probs}}}
        )

    model = JevosDecisionModel("http://jev.test")
    model._client_obj = httpx.AsyncClient(
        base_url="http://jev.test", transport=httpx.MockTransport(handler)
    )
    d = await model.decide(CV, CHOICE)
    assert d.backend == "jevos" and model.model_name == "jevos-v5"
    assert list(d.probs) == CHOICE.option_labels and _sums_to_one(d.probs)
    assert seen[0]["state"] == CV and seen[0]["questions"]["q"]["type"] == "choice"
    await model.aclose()


async def test_jevos_without_server_or_binary_raises_clearly(tmp_path) -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    model = JevosDecisionModel("http://jev.test", binary=tmp_path / "missing-jev")
    model._client_obj = httpx.AsyncClient(
        base_url="http://jev.test", transport=httpx.MockTransport(down)
    )
    with pytest.raises(RuntimeError, match="no `jev` binary"):
        await model.decide(CV, YES_NO)
    await model.aclose()


# --- real models (downloads / local binary; skipped in CI) -----------------------------------


@pytest.mark.llm
async def test_laya_real_model() -> None:
    pytest.importorskip("laya")
    model = LayaDecisionModel("multilingual")
    d = await model.decide(CV, YES_NO)
    assert _sums_to_one(d.probs) and d.answer == "yes"
    fr = await model.decide(
        "Marie Dupont, développeuse, 7 ans d'expérience, Docker et Kubernetes en production.",
        Question(kind="yes_no", question="Ce CV montre-t-il une expérience de Docker ?"),
    )
    # zero-shot accuracy is the benchmark's job (bench/decisions); here only the plumbing
    assert list(fr.probs) == ["no", "yes"] and _sums_to_one(fr.probs)
    await model.aclose()


@pytest.mark.llm
async def test_jevos_real_model() -> None:
    if not DEFAULT_BINARY.is_file():
        pytest.skip("jev binary not installed (see bench/decisions/BACKENDS.md)")
    model = JevosDecisionModel()
    try:
        d = await model.decide(CV, YES_NO)
        assert _sums_to_one(d.probs) and d.answer == "yes"
        c = await model.decide(CV, CHOICE)
        assert c.answer == "senior"
    finally:
        await model.aclose()
