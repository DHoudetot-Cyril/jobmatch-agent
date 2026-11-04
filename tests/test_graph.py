"""Graph tests with a scripted fake LLM (no llama-server needed)."""

import time
import uuid

import pytest
from conftest import INVALID, SAMPLES
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END
from langgraph.types import Command

from jobmatch.config import get_settings
from jobmatch.graph.build import build_graph
from jobmatch.graph.structured import StructuredOutputError
from jobmatch.graph.supervisor import next_step
from jobmatch.graph.teams.matching import (
    GapAnalysis,
    ScoreAssessment,
    SkillAssessment,
    compute_score,
)
from jobmatch.schemas import (
    AnalysisStatus,
    CandidateProfile,
    CoverLetter,
    HumanDecision,
    JobOffer,
    MatchResult,
    Review,
    SkillLevel,
)

INPUT = {"offer_text": "Offre Data Engineer ...", "cv_text": "CV de Camille ..."}


@pytest.fixture
def assessment() -> ScoreAssessment:
    return ScoreAssessment(
        skills=[
            SkillAssessment(skill="Python", level="required", match="full", evidence="CV"),
            SkillAssessment(skill="Apache Spark", level="required", match="full", evidence="CV"),
            # wrong level on purpose: the offer's level must win
            SkillAssessment(skill="Kafka", level="required", match="none", evidence="absent"),
        ],
        experience_fit=80,
        strengths=["PySpark à grande échelle"],
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


@pytest.fixture
def graph():
    return build_graph(checkpointer=MemorySaver())


def _config() -> dict:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


async def _resume(graph, config, **decision) -> dict:
    return await graph.ainvoke(Command(resume=HumanDecision(**decision).model_dump()), config)


async def test_happy_path_pauses_at_human_review(graph, install_fake_llm, responses):
    fake = install_fake_llm(responses)
    config = _config()
    result = await graph.ainvoke(INPUT, config)

    [intr] = result["__interrupt__"]
    assert set(intr.value) == {"cover_letter", "match"}
    assert CoverLetter.model_validate(intr.value["cover_letter"]).language == "fr"
    match = MatchResult.model_validate(intr.value["match"])
    # required 2/2 * 70 + nice 0/1 * 15 + 0.8 * 15 = 82
    assert match.score == 82
    assert match.matched_skills == ["Python", "Apache Spark"]
    assert [(g.skill, g.level) for g in match.gaps] == [("Kafka", SkillLevel.nice_to_have)]

    state = (await graph.aget_state(config)).values
    assert state["status"] == AnalysisStatus.awaiting_approval
    assert state["review_iterations"] == 1
    assert state["trace"][0] == "supervisor"
    assert sorted(state["trace"][1:3]) == ["parse_cv", "parse_offer"]  # parallel fan-out
    assert state["trace"][3:] == [
        "supervisor",
        "score_match",
        "verify_skills",
        "analyze_gaps",
        "supervisor",
        "write_letter",
        "review_letter",
        "verify_claims",
        "supervisor",
    ]
    assert fake.count(JobOffer) == fake.count(CandidateProfile) == 1


async def test_resume_approved_completes(graph, install_fake_llm, responses):
    install_fake_llm(responses)
    config = _config()
    await graph.ainvoke(INPUT, config)
    result = await _resume(graph, config, approved=True)
    assert result["status"] == AnalysisStatus.completed
    assert "__interrupt__" not in result
    assert result["trace"][-2:] == ["human_review", "supervisor"]
    assert not (await graph.aget_state(config)).next  # graph finished


async def test_resume_with_feedback_rewrites_then_pauses_again(
    graph, install_fake_llm, responses, letter
):
    rewritten = letter.model_copy(update={"body": "Version plus courte"})
    responses[CoverLetter] = [letter, rewritten]
    fake = install_fake_llm(responses)
    config = _config()
    await graph.ainvoke(INPUT, config)

    result = await _resume(graph, config, approved=False, feedback="Plus court svp")
    [intr] = result["__interrupt__"]
    assert intr.value["cover_letter"]["body"] == "Version plus courte"
    state = (await graph.aget_state(config)).values
    assert state["status"] == AnalysisStatus.awaiting_approval
    assert state["human_feedback"] == "Plus court svp"
    assert state["review_iterations"] == 1  # fresh budget for the rewrite round
    # The writer received the previous draft and the feedback; analysis did not re-run.
    rewrite_prompt = fake.calls["CoverLetter"][1][-1].content
    assert "Plus court svp" in rewrite_prompt and letter.body in rewrite_prompt
    assert "Plus court svp" in fake.calls["Review"][1][0].content
    assert fake.count(JobOffer) == 1 and fake.count(ScoreAssessment) == 1

    result = await _resume(graph, config, approved=True)
    assert result["status"] == AnalysisStatus.completed


async def test_resume_rejected_without_feedback(graph, install_fake_llm, responses):
    fake = install_fake_llm(responses)
    config = _config()
    await graph.ainvoke(INPUT, config)
    result = await _resume(graph, config, approved=False, feedback="  ")
    assert result["status"] == AnalysisStatus.rejected
    assert not (await graph.aget_state(config)).next
    assert fake.count(CoverLetter) == 1  # no rewrite


async def test_review_loop_stops_at_max_iterations(
    graph, install_fake_llm, responses, rejected_review
):
    responses[Review] = [rejected_review]  # the reviewer never approves
    fake = install_fake_llm(responses)
    config = _config()
    result = await graph.ainvoke(INPUT, config)

    max_iter = get_settings().max_review_iterations
    assert "__interrupt__" in result  # still handed to the human
    state = (await graph.aget_state(config)).values
    assert state["review_iterations"] == max_iter
    assert fake.count(CoverLetter) == fake.count(Review) == max_iter
    # The second draft was asked to fix the reviewer's issues.
    assert rejected_review.issues[0] in fake.calls["CoverLetter"][1][-1].content


async def test_reviewer_approval_with_issues_is_not_approved(graph, install_fake_llm, responses):
    responses[Review] = [Review(approved=True, issues=["Invented certification"])]
    install_fake_llm(responses)
    config = _config()
    await graph.ainvoke(INPUT, config)
    state = (await graph.aget_state(config)).values
    assert state["review"].approved is False
    assert state["review_iterations"] == get_settings().max_review_iterations


async def test_parsing_failure_retries_once(graph, install_fake_llm, responses, offer):
    responses[JobOffer] = [INVALID, offer]
    fake = install_fake_llm(responses)
    result = await graph.ainvoke(INPUT, _config())
    assert "__interrupt__" in result
    assert fake.count(JobOffer) == 2
    assert "could not be parsed" in fake.calls["JobOffer"][1][-1].content


async def test_parsing_failure_twice_raises(graph, install_fake_llm, responses):
    responses[CandidateProfile] = [INVALID]
    install_fake_llm(responses)
    with pytest.raises(StructuredOutputError, match="parse_cv.*CandidateProfile"):
        await graph.ainvoke(INPUT, _config())


def test_compute_score_rubric():
    def a(*matches, fit=100):
        return ScoreAssessment(
            skills=[
                SkillAssessment(skill=f"s{i}", level=lvl, match=m, evidence="")
                for i, (lvl, m) in enumerate(matches)
            ],
            experience_fit=fit,
            summary="",
        )

    assert compute_score(a(("required", "full"), ("nice_to_have", "full"))) == 100
    assert compute_score(a(("required", "none"), ("nice_to_have", "full"), fit=0)) == 15
    assert compute_score(a(("required", "full"), ("nice_to_have", "none"), fit=0)) == 70
    assert compute_score(a(("required", "partial"), ("nice_to_have", "none"), fit=0)) == 35
    # no nice-to-have skills: weights are renormalised over required + experience
    assert compute_score(a(("required", "full"), fit=0)) == round(100 * 70 / 85)
    assert compute_score(a(fit=60)) == 60


def test_supervisor_routing(offer, profile, letter, approved_review):
    match = MatchResult(score=50, summary="ok")
    assert next_step({}) == "analysis_team"
    assert next_step({"offer": offer}) == "analysis_team"
    assert next_step({"offer": offer, "profile": profile}) == "matching_team"
    base = {"offer": offer, "profile": profile, "match": match}
    assert next_step(base) == "writing_team"
    assert next_step({**base, "cover_letter": letter}) == "writing_team"
    full = {**base, "cover_letter": letter, "review": approved_review}
    assert next_step(full) == "human_review"
    assert next_step({**full, "review": None, "human_feedback": "x"}) == "writing_team"
    assert next_step({**full, "status": AnalysisStatus.completed}) == END
    assert next_step({**full, "status": AnalysisStatus.rejected}) == END
    assert next_step({**full, "status": AnalysisStatus.running}) == "human_review"


def test_graph_structure():
    g = build_graph().get_graph()
    assert {"supervisor", "analysis_team", "matching_team", "writing_team", "human_review"} <= set(
        g.nodes
    )


@pytest.mark.llm
async def test_real_llm_end_to_end():
    """Full run against llama-server on the fictional samples, up to the human review."""
    graph = build_graph(checkpointer=MemorySaver())
    config = _config()
    inputs = {
        "offer_text": (SAMPLES / "offer_data_engineer.txt").read_text(),
        "cv_text": (SAMPLES / "cv_data_engineer.txt").read_text(),
    }
    start = last = time.perf_counter()
    timings: dict[str, float] = {}  # seconds per top-level node (teams)
    async for update in graph.astream(inputs, config, stream_mode="updates"):
        now = time.perf_counter()
        for node in update:
            timings[node] = timings.get(node, 0.0) + now - last
        last = now
    elapsed = time.perf_counter() - start

    snapshot = await graph.aget_state(config)
    [intr] = snapshot.interrupts
    match = MatchResult.model_validate(intr.value["match"])
    letter = CoverLetter.model_validate(intr.value["cover_letter"])
    state = snapshot.values
    print(f"\nE2E {elapsed:.0f}s", {k: round(v, 1) for k, v in timings.items()})
    print(f"score={match.score} iterations={state['review_iterations']}")
    print("matched:", match.matched_skills, "| gaps:", [g.skill for g in match.gaps])
    print("review:", state["review"])
    print(letter.subject, "\n", letter.body)

    assert state["status"] == AnalysisStatus.awaiting_approval
    assert state["offer"].language == "fr"
    assert "Camille" in (state["profile"].name or "")
    assert 55 <= match.score <= 100  # strong but not perfect fit (no Kafka / dbt / Terraform)
    matched = " ".join(match.matched_skills).casefold()
    assert "python" in matched and "sql" in matched
    assert match.gaps and all(g.advice for g in match.gaps)
    assert letter.language == "fr" and len(letter.body) > 500


def test_merge_advice_keeps_canonical_gap_names(gap):
    from jobmatch.graph.teams.matching import _merge_advice

    advised = [gap.model_copy(update={"skill": "Kafka (nice_to_have)", "advice": "Projet Kafka"})]
    [merged] = _merge_advice([gap], advised)
    assert (merged.skill, merged.level, merged.advice) == ("Kafka", gap.level, "Projet Kafka")
    assert _merge_advice([gap], [])[0].advice == gap.advice
