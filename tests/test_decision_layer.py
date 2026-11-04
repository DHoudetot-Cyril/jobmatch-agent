"""Decision layer in the graph (verify_skills / verify_claims) with a FAKE decision model."""

import json
import sys
import uuid
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver

from jobmatch.config import get_settings
from jobmatch.decisions import service
from jobmatch.graph.build import build_graph
from jobmatch.graph.teams.matching import (
    WEIGHT_EXPERIENCE,
    WEIGHT_NICE_TO_HAVE,
    WEIGHT_REQUIRED,
    GapAnalysis,
    ScoreAssessment,
    SkillAssessment,
)
from jobmatch.graph.teams.writing import _merge_claims, claim_issue
from jobmatch.observability.metrics import CLAIMS_DISMISSED, DECISIONS
from jobmatch.schemas import (
    CandidateProfile,
    CoverLetter,
    JobOffer,
    Review,
    ReviewerOutput,
    SkillCheck,
    SkillLevel,
)

BENCH = Path(__file__).resolve().parents[1] / "bench" / "decisions"
sys.path.insert(0, str(BENCH))
import build_dataset  # noqa: E402
import metrics as bench_metrics  # noqa: E402

INPUT = {"offer_text": "Offre Data Engineer ...", "cv_text": "CV de Camille ..."}
TRUE_CLAIM = "J'ai trois ans d'expérience avec Spark"
FALSE_CLAIM = "J'ai dirigé une équipe de dix personnes"


@pytest.fixture
def assessment() -> ScoreAssessment:
    return ScoreAssessment(
        skills=[
            SkillAssessment(skill="Python", level="required", match="full", evidence="CV"),
            SkillAssessment(skill="Apache Spark", level="required", match="partial", evidence="?"),
            SkillAssessment(skill="Kafka", level="nice_to_have", match="none", evidence="absent"),
        ],
        experience_fit=80,
        summary="Bon profil.",
    )


@pytest.fixture
def responses(offer, profile, assessment, gap, letter, approved_review) -> dict:
    return {
        JobOffer: [offer],
        CandidateProfile: [profile],
        ScoreAssessment: [assessment],
        GapAnalysis: [GapAnalysis(gaps=[gap])],
        CoverLetter: [letter],
        Review: [approved_review],
    }


def by_keyword(table: dict[str, float]):
    """Fake p_yes: the first keyword of `table` found in the question."""
    return lambda state, q: next(p for k, p in table.items() if k in q)


async def run(responses, install_fake_llm) -> dict:
    fake = install_fake_llm(responses)
    graph = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    await graph.ainvoke(INPUT, config)
    return {"state": (await graph.aget_state(config)).values, "llm": fake}


def counter(metric, **labels) -> float:
    return metric.labels(**labels)._value.get() if labels else metric._value.get()


# ------------------------------------------------------------------ bench fidelity / math


def test_templates_and_temperatures_come_from_the_benchmark():
    assert build_dataset.SKILL_Q["en"] == service.SKILL_QUESTION
    assert build_dataset.CLAIM_Q["en"] == service.CLAIM_QUESTION
    summary = json.loads((BENCH / "results" / "summary.json").read_text())
    for task, t in service.TEMPERATURE.items():
        assert t == pytest.approx(summary["jevos@q-en"]["metrics"][task]["temperature"])


@pytest.mark.parametrize("p", [0.01, 0.3, 0.5, 0.9, 0.999])
def test_calibration_matches_bench_formula(p):
    t = service.TEMPERATURE["skill_evidence"]
    expected = bench_metrics.apply_temperature({"no": 1 - p, "yes": p}, t)["yes"]
    assert service.apply_temperature(p, t) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("match", "p", "final", "resolution", "agree"),
    [
        ("partial", 0.8, 0.8, "resolved_partial", True),  # unsure LLM: System 1 decides
        ("partial", 0.1, 0.1, "resolved_partial", False),
        ("full", 0.6, 1.0, "llm", True),  # no systematic down-bias (v1 gave 0.8)
        ("full", 0.3, 1.0, "llm", False),  # mild disagreement: LLM kept
        ("full", 0.1, 0.5, "conflict", False),  # strong disagreement: partial credit
        ("none", 0.5, 0.0, "llm", False),
        ("none", 0.9, 0.5, "conflict", False),
        ("none", 0.05, 0.0, "llm", True),
    ],
)
def test_fusion_v2(match, p, final, resolution, agree):
    a = SkillAssessment(skill="Docker", level="required", match=match, evidence="")
    c = service.fuse_skill(a, p)
    assert (c.final_credit, c.resolution, c.agree) == (final, resolution, agree)
    assert c.llm_credit == service.LLM_CREDIT[match] and c.p_evidence == p


def test_fusion_thresholds_come_from_settings(monkeypatch):
    a = SkillAssessment(skill="Docker", level="required", match="full", evidence="")
    assert service.fuse_skill(a, 0.3).resolution == "llm"
    monkeypatch.setattr(get_settings(), "decision_conflict_low", 0.4)
    assert service.fuse_skill(a, 0.3).resolution == "conflict"
    monkeypatch.setattr(get_settings(), "decision_conflict_high", 0.95)
    b = a.model_copy(update={"match": "none"})
    assert service.fuse_skill(b, 0.9).resolution == "llm"


def test_skill_check_resolution_defaults_for_old_outputs():
    old = {"skill": "Go", "level": "required", "llm_credit": 1.0, "p_evidence": 0.9}
    assert SkillCheck.model_validate({**old, "final_credit": 0.95, "agree": True}).resolution == (
        "llm"
    )


def test_non_jevos_backend_is_uncalibrated(monkeypatch, caplog):
    monkeypatch.setattr(service, "_warned_uncalibrated", False)
    assert (
        service.calibration_temperature("skill_evidence") == service.TEMPERATURE["skill_evidence"]
    )
    monkeypatch.setattr(get_settings(), "decisions_backend", "qwen-logits")
    with caplog.at_level("WARNING"):
        assert service.calibration_temperature("skill_evidence") == 1.0
        assert service.calibration_temperature("claim_supported") == 1.0
    assert sum("no calibration temperature" in r.message for r in caplog.records) == 1


async def test_non_jevos_backend_returns_raw_probabilities(install_fake_decisions, monkeypatch):
    monkeypatch.setattr(get_settings(), "decisions_backend", "laya")
    install_fake_decisions(lambda state, q: 0.7)
    a = SkillAssessment(skill="Go", level="required", match="partial", evidence="")
    (check,) = await service.check_skill_evidence("CV", [a])
    assert check.p_evidence == pytest.approx(0.7)


# ------------------------------------------------------------------ verify_skills


async def test_verify_skills_fuses_and_recomputes(
    install_fake_llm, install_fake_decisions, responses
):
    fake = install_fake_decisions(by_keyword({"Python": 0.9, "Spark": 0.9, "Kafka": 0.2}))
    state = (await run(responses, install_fake_llm))["state"]
    match = state["match"]

    cal = {
        k: service.apply_temperature(p, service.TEMPERATURE["skill_evidence"])
        for k, p in {"hi": 0.9, "lo": 0.2}.items()
    }
    checks = {c.skill: c for c in match.skill_checks}
    assert checks["Python"].final_credit == 1.0  # "full" + evidence: LLM kept
    assert checks["Apache Spark"].final_credit == pytest.approx(cal["hi"], abs=1e-4)
    assert checks["Kafka"].final_credit == 0.0  # "none" + low evidence: LLM kept
    assert [c.resolution for c in match.skill_checks] == ["llm", "resolved_partial", "llm"]
    assert [c.agree for c in match.skill_checks] == [True, True, True]
    expected = round(
        100
        * (
            WEIGHT_REQUIRED
            * (checks["Python"].final_credit + checks["Apache Spark"].final_credit)
            / 2
            + WEIGHT_NICE_TO_HAVE * checks["Kafka"].final_credit
            + WEIGHT_EXPERIENCE * 0.8
        )
    )
    assert match.score == expected == 80  # LLM-only: (1 + 0.5) / 2 * 70 + 0 + 12 = 64.5 -> 64
    # the partial skill is confirmed by the decision model: matched, no longer a gap
    assert match.matched_skills == ["Python", "Apache Spark"]
    assert [g.skill for g in match.gaps] == ["Kafka"]
    assert len(fake.questions) == 3 and all(
        q.startswith("Does the CV show") for q in fake.questions
    )


async def test_verify_skills_low_evidence_turns_partial_into_gap(
    install_fake_llm, install_fake_decisions, responses
):
    install_fake_decisions(by_keyword({"Python": 0.9, "Spark": 0.05, "Kafka": 0.05}))
    match = (await run(responses, install_fake_llm))["state"]["match"]
    assert match.matched_skills == ["Python"]
    assert [g.skill for g in match.gaps] == ["Apache Spark", "Kafka"]
    spark = next(c for c in match.skill_checks if c.skill == "Apache Spark")
    assert spark.final_credit < 0.5 and spark.agree is False


# ------------------------------------------------------------------ degradation / disabled


def _outputs(state: dict) -> dict:
    r = state["review"]
    return {
        "match": state["match"].model_dump(),
        "review": (r.approved, r.issues, r.suggestions),
        "letter": state["cover_letter"].model_dump(),
        "iterations": state["review_iterations"],
    }


@pytest.mark.parametrize("reviewer", ["approved_review", "rejected_review"])
async def test_backend_failure_gives_llm_only_results(
    request, install_fake_llm, install_fake_decisions, responses, monkeypatch, reviewer
):
    responses[Review] = [request.getfixturevalue(reviewer)]
    install_fake_decisions(error=ConnectionError("jevos down"))
    before = counter(DECISIONS, task="skill_evidence", outcome="error")
    degraded = await run(responses, install_fake_llm)
    assert counter(DECISIONS, task="skill_evidence", outcome="error") == before + 1

    monkeypatch.setattr(get_settings(), "decisions_enabled", False)
    legacy = await run(responses, install_fake_llm)
    assert _outputs(degraded["state"]) == _outputs(legacy["state"])
    assert degraded["state"]["match"].skill_checks == []


async def test_disabled_is_the_historical_pipeline(
    install_fake_llm, install_fake_decisions, responses, monkeypatch
):
    monkeypatch.setattr(get_settings(), "decisions_enabled", False)
    fake = install_fake_decisions(lambda s, q: 0.99)
    result = await run(responses, install_fake_llm)
    state, llm = result["state"], result["llm"]
    assert fake.questions == []  # the decision model is never called
    assert ReviewerOutput not in llm.schemas and Review in llm.schemas
    system = llm.calls["Review"][0][0].content
    assert "list in `issues` EVERY claim" in system and "unsupported_claims" not in system
    assert state["match"].score == 64  # LLM-only rubric
    assert state["match"].skill_checks == [] and state["review"].claim_checks == []


# ------------------------------------------------------------------ verify_claims


async def test_verify_claims_dismisses_supported_claims(
    install_fake_llm, install_fake_decisions, responses
):
    responses[Review] = [
        Review(
            approved=False,
            issues=["Letter is too long", f"The claim « {TRUE_CLAIM} » is not in the CV"],
            unsupported_claims=[TRUE_CLAIM, FALSE_CLAIM],
        )
    ]
    install_fake_decisions(by_keyword({"trois ans": 0.97, "dirigé": 0.1, "": 0.9}))
    before = counter(CLAIMS_DISMISSED)
    result = await run(responses, install_fake_llm)
    review = result["state"]["review"]

    assert counter(CLAIMS_DISMISSED) == before + get_settings().max_review_iterations
    assert [(c.claim, c.dismissed) for c in review.claim_checks] == [
        (TRUE_CLAIM, True),
        (FALSE_CLAIM, False),
    ]
    # the LLM's duplicate of the claim was merged; the dismissed claim's issue is gone
    assert review.issues == ["Letter is too long", claim_issue(FALSE_CLAIM)]
    assert review.unsupported_claims == [FALSE_CLAIM]
    assert review.approved is False
    # the writer is asked to fix only the remaining issues
    rewrite = result["llm"].calls["CoverLetter"][1][-1].content
    assert FALSE_CLAIM in rewrite and TRUE_CLAIM not in rewrite


async def test_verify_claims_approves_when_all_issues_are_dismissed(
    install_fake_llm, install_fake_decisions, responses
):
    responses[Review] = [Review(approved=False, unsupported_claims=[TRUE_CLAIM])]
    install_fake_decisions(by_keyword({"trois ans": 0.97, "": 0.9}))
    result = await run(responses, install_fake_llm)
    state = result["state"]
    assert state["review"].approved is True and state["review"].issues == []
    assert state["review_iterations"] == 1 and result["llm"].count(CoverLetter) == 1
    assert state["trace"][-3:] == ["review_letter", "verify_claims", "supervisor"]


async def test_claim_threshold_is_calibrated_probability(install_fake_decisions):
    t = service.TEMPERATURE["claim_supported"]
    install_fake_decisions(lambda s, q: 0.8)
    [c] = await service.check_claims("cv", ["x"])
    assert c.p_supported == pytest.approx(service.apply_temperature(0.8, t), abs=1e-4)
    assert c.dismissed is (c.p_supported >= 0.8)


def test_merge_claims_keeps_other_issues():
    out = ReviewerOutput(approved=True, issues=["Wrong language"], unsupported_claims=[" x ", ""])
    r = _merge_claims(out)
    assert r.issues == ["Wrong language", claim_issue("x")] and r.approved is False
    assert type(r) is Review and r.unsupported_claims == ["x"]


# ------------------------------------------------------------------ service edge cases


async def test_too_long_state_is_skipped(install_fake_decisions):
    fake = install_fake_decisions(lambda s, q: 0.9)
    before = counter(DECISIONS, task="claim_supported", outcome="skipped")
    assert await service.check_claims("x" * (service.MAX_STATE_CHARS + 1), ["claim"]) == []
    assert fake.questions == []
    assert counter(DECISIONS, task="claim_supported", outcome="skipped") == before + 1


async def test_http_422_is_a_skip_not_an_error(install_fake_decisions):
    install_fake_decisions(error=RuntimeError("jevos: HTTP 422: state too long"))
    before = counter(DECISIONS, task="skill_evidence", outcome="skipped")
    a = SkillAssessment(skill="Go", level=SkillLevel.required, match="none", evidence="")
    assert await service.check_skill_evidence("cv", [a]) == []
    assert counter(DECISIONS, task="skill_evidence", outcome="skipped") == before + 1
