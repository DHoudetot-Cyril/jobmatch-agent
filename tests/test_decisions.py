"""Decision layer: base contract, registry, benchmark metrics, qwen-logits (mocked HTTP)."""

import json
import math
import sys
from pathlib import Path

import httpx
import pytest

from jobmatch.decisions.base import Decision, Question
from jobmatch.decisions.qwen_logits import (
    DecisionError,
    QwenLogitsDecisionModel,
    build_messages,
    parse_letter_probs,
)
from jobmatch.decisions.registry import available_backends, canonical_name, get_backend

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench" / "decisions"))
import metrics as M  # noqa: E402

# ----------------------------------------------------------------------------- contract


def test_yes_no_options_are_fixed():
    q = Question(kind="yes_no", question="?", options={"a": "x"})
    assert q.option_labels == ["no", "yes"]


def test_choice_needs_2_to_16_options():
    with pytest.raises(ValueError):
        Question(kind="choice", question="?", options={"a": "x"})
    with pytest.raises(ValueError):
        Question(kind="choice", question="?", options={str(i): "" for i in range(17)})
    assert Question(kind="choice", question="?", options={"a": "", "b": ""}).option_labels == [
        "a",
        "b",
    ]


def test_decision_answer_and_confidence():
    d = Decision(probs={"no": 0.2, "yes": 0.8})
    assert d.answer == "yes"
    assert d.confidence == pytest.approx(0.8)


# ----------------------------------------------------------------------------- registry


def test_registry_aliases_and_unknown():
    assert canonical_name("Qwen") == "qwen-logits"
    assert canonical_name("qwen-perm") == "qwen-logits-perm"
    assert canonical_name("jev") == "jevos"
    with pytest.raises(KeyError):
        canonical_name("nope")


def test_registry_builds_qwen_lazily():
    m = get_backend("qwen-logits-perm", base_url="http://x/v1", model="m")
    assert isinstance(m, QwenLogitsDecisionModel)
    assert m.permute and m.name == "qwen-logits-perm"
    avail = available_backends()
    assert avail["qwen-logits"] is True
    assert set(avail) >= {"qwen-logits", "laya", "jevos"}


# ----------------------------------------------------------------------------- metrics


def test_accuracy_f1_brier():
    preds = [{"no": 0.9, "yes": 0.1}, {"no": 0.2, "yes": 0.8}, {"no": 0.6, "yes": 0.4}]
    gold = ["no", "yes", "yes"]
    assert M.accuracy(preds, gold) == pytest.approx(2 / 3)
    # no: tp1 fp1 fn0 -> 2/3 ; yes: tp1 fp0 fn1 -> 2/3
    assert M.macro_f1(preds, gold, ["no", "yes"]) == pytest.approx(2 / 3)
    expected = ((0.1**2 * 2) + (0.2**2 * 2) + (0.6**2 * 2)) / 3
    assert M.brier(preds, gold) == pytest.approx(expected)


def test_ece_perfect_and_overconfident():
    # confidence 0.8 with 80 % accuracy -> perfectly calibrated
    preds = [{"no": 0.2, "yes": 0.8}] * 10
    gold = ["yes"] * 8 + ["no"] * 2
    assert M.ece(preds, gold) == pytest.approx(0.0, abs=1e-9)
    # confidence 1.0 with 50 % accuracy -> ECE 0.5
    preds = [{"no": 0.0, "yes": 1.0}] * 4
    assert M.ece(preds, ["yes", "no", "yes", "no"]) == pytest.approx(0.5)


def test_temperature_scaling_softens_overconfidence():
    preds = [{"no": 0.01, "yes": 0.99}] * 6 + [{"no": 0.99, "yes": 0.01}] * 4
    gold = ["yes"] * 4 + ["no"] * 2 + ["no"] * 3 + ["yes"]  # 70 % accurate, 99 % confident
    t = M.fit_temperature(preds, gold)
    assert t > 1.5
    cal = M.apply_temperature(preds[0], t)
    assert cal["yes"] == pytest.approx(0.7, abs=0.02)
    # argmax is preserved, T=1 is identity
    assert M.argmax(cal) == "yes"
    assert M.apply_temperature(preds[0], 1.0)["yes"] == pytest.approx(0.99)


def test_automatable_share():
    preds = [{"no": 1 - c, "yes": c} for c in (0.99, 0.95, 0.9, 0.8, 0.7)]
    gold = ["yes", "yes", "yes", "no", "yes"]
    # top 3 correct; adding the 4th (wrong) -> 25 % errors
    assert M.automatable_share(preds, gold, 0.05) == pytest.approx(3 / 5)
    assert M.automatable_share(preds, gold, 0.25) == pytest.approx(1.0)  # 1/5 = 20 % <= 25 %
    # ties are answered together
    tied = [{"no": 0.1, "yes": 0.9}] * 2
    assert M.automatable_share(tied, ["yes", "no"], 0.05) == 0.0


def test_percentile():
    assert M.percentile([1, 2, 3, 4], 0.5) == pytest.approx(2.5)
    assert math.isnan(M.percentile([], 0.5))


# ----------------------------------------------------------------------------- qwen-logits


def _resp(top: list[tuple[str, float]]) -> dict:
    return {
        "choices": [
            {
                "logprobs": {
                    "content": [{"top_logprobs": [{"token": t, "logprob": lp} for t, lp in top]}]
                }
            }
        ]
    }


def test_build_messages_letters_and_order():
    q = Question(kind="choice", question="Type?", options={"cv": "A CV", "job_offer": "An offer"})
    msgs = build_messages("DOC", q, ["job_offer", "cv"])
    user = msgs[-1]["content"]
    assert user.startswith("<document>\nDOC\n</document>")
    assert "A. job_offer — An offer" in user and "B. cv — A CV" in user
    assert "(A, B)" in user


def test_parse_merges_variants_and_floors_missing():
    resp = _resp([("A", math.log(0.6)), (" A", math.log(0.2)), ("Yes", math.log(0.1))])
    p = parse_letter_probs(resp, 2)
    # A = 0.8, B missing -> floor = min logprob = 0.1
    assert p[0] == pytest.approx(0.8 / 0.9)
    assert p[1] == pytest.approx(0.1 / 0.9)


def test_parse_raises_without_letters():
    with pytest.raises(DecisionError):
        parse_letter_probs(_resp([("Hello", -0.1)]), 2)
    with pytest.raises(DecisionError):
        parse_letter_probs({"choices": [{}]}, 2)


async def test_decide_with_mock_transport_and_permutation():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        user = body["messages"][-1]["content"]
        # model always prefers letter A (position bias): P(A)=0.7
        a_label = "no" if "A. no" in user else "yes"
        assert a_label in ("no", "yes")
        return httpx.Response(200, json=_resp([("A", math.log(0.7)), ("B", math.log(0.3))]))

    model = QwenLogitsDecisionModel(
        "http://llm/v1", "qwen", permute=False, transport=httpx.MockTransport(handler)
    )
    q = Question(kind="yes_no", question="Docker?")
    d = await model.decide("CV text", q)
    assert d.probs == pytest.approx({"no": 0.7, "yes": 0.3})
    assert d.backend == "qwen-logits"
    body = seen[0]
    assert body["max_tokens"] == 1 and body["logprobs"] is True
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen[0]["model"] == "qwen"
    await model.aclose()

    # permutation averaging cancels a pure position bias
    seen.clear()
    perm = QwenLogitsDecisionModel(
        "http://llm/v1", "qwen", permute=True, transport=httpx.MockTransport(handler)
    )
    d = await perm.decide("CV text", q)
    assert len(seen) == 2
    assert d.probs == pytest.approx({"no": 0.5, "yes": 0.5})
    assert d.backend == "qwen-logits-perm"
    await perm.aclose()


async def test_decide_retries_on_5xx():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="loading")
        return httpx.Response(200, json=_resp([("B", -0.1), ("A", -3.0)]))

    model = QwenLogitsDecisionModel(
        "http://llm/v1", "qwen", retries=2, transport=httpx.MockTransport(handler)
    )
    model_sleep = pytest.MonkeyPatch()
    import jobmatch.decisions.qwen_logits as ql

    async def no_sleep(_s):
        return None

    model_sleep.setattr(ql.asyncio, "sleep", no_sleep)
    try:
        d = await model.decide("x", Question(kind="yes_no", question="?"))
    finally:
        model_sleep.undo()
    assert calls["n"] == 2
    assert d.answer == "yes"
    await model.aclose()


def test_wilson_low():
    assert M.wilson_low(1.0, 7) == pytest.approx(0.646, abs=0.01)
    assert M.wilson_low(0.5, 1000) == pytest.approx(0.469, abs=0.005)
