"""Decision layer v2 fixes: duplicate offer skills, control-flow node errors, reviewer token
budget, length errors, /ready decisions check, OpenAPI review fields. No LLM needed."""

import uuid

import openai
import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphInterrupt
from openai.types.chat import ChatCompletion
from openai.types.completion_usage import CompletionUsage

from jobmatch import llm
from jobmatch.config import get_settings
from jobmatch.graph.build import build_graph
from jobmatch.graph.structured import StructuredOutputError, invoke_structured
from jobmatch.graph.teams.analysis import dedupe_skills
from jobmatch.graph.teams.matching import GapAnalysis
from jobmatch.observability.metrics import NODE_ERRORS, SKILL_RESOLUTION, instrument_node
from jobmatch.schemas import (
    AnalysisStatus,
    CandidateProfile,
    CoverLetter,
    JobOffer,
    Review,
    ScoreAssessment,
    Skill,
    SkillAssessment,
    SkillLevel,
)

INPUT = {"offer_text": "Offre Cloud Engineer ...", "cv_text": "CV ..."}


def counter(metric, **labels) -> float:
    return metric.labels(**labels)._value.get()


# ------------------------------------------------------------------ 3a duplicate offer skills


def test_dedupe_skills_required_wins_and_keeps_order():
    skills = [
        Skill(name="AWS", level=SkillLevel.nice_to_have),
        Skill(name="Python", level=SkillLevel.required),
        Skill(name="aws ", level=SkillLevel.required),
        Skill(name="Python", level=SkillLevel.nice_to_have),
        Skill(name="  ", level=SkillLevel.required),
    ]
    assert [(s.name, s.level) for s in dedupe_skills(skills)] == [
        ("AWS", SkillLevel.required),
        ("Python", SkillLevel.required),
    ]


@pytest.fixture
def dup_responses(profile, letter) -> dict:
    offer = JobOffer(
        title="Cloud Engineer",
        skills=[
            Skill(name="AWS", level=SkillLevel.required),
            Skill(name="Terraform", level=SkillLevel.required),
            Skill(name="AWS", level=SkillLevel.nice_to_have),  # duplicate, weaker level LAST
        ],
    )
    assessment = ScoreAssessment(
        skills=[
            SkillAssessment(skill="AWS", level="nice_to_have", match="none", evidence="-"),
            SkillAssessment(skill="Terraform", level="required", match="full", evidence="CV"),
            SkillAssessment(skill="aws", level="nice_to_have", match="full", evidence="dup"),
        ],
        experience_fit=50,
        summary="ok",
    )
    return {
        JobOffer: [offer],
        CandidateProfile: [profile],
        ScoreAssessment: [assessment],
        GapAnalysis: [GapAnalysis()],
        CoverLetter: [letter],
        Review: [Review(approved=True)],
    }


async def test_duplicate_offer_skill_stays_required(install_fake_llm, dup_responses):
    graph = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    install_fake_llm(dup_responses)
    await graph.ainvoke(INPUT, config)
    state = (await graph.aget_state(config)).values
    assert [(s.name, s.level) for s in state["offer"].skills] == [
        ("AWS", SkillLevel.required),
        ("Terraform", SkillLevel.required),
    ]
    assert [(s.skill, s.level) for s in state["score_assessment"].skills] == [
        ("AWS", SkillLevel.required),
        ("Terraform", SkillLevel.required),
    ]
    match = state["match"]
    assert [(g.skill, g.level) for g in match.gaps] == [("AWS", SkillLevel.required)]
    assert match.score == round(100 * (0.70 * 0.5 + 0.15 * 0.5) / 0.85)  # 50


# ------------------------------------------------------------------ 3b interrupts aren't errors


async def test_instrument_node_ignores_graph_interrupt():
    @instrument_node("human_review")
    async def paused(_state):
        raise GraphInterrupt(())

    @instrument_node("human_review")
    def paused_sync(_state):
        raise GraphInterrupt(())

    before = counter(NODE_ERRORS, node="human_review")
    with pytest.raises(GraphInterrupt):
        await paused({})
    with pytest.raises(GraphInterrupt):
        paused_sync({})
    assert counter(NODE_ERRORS, node="human_review") == before


async def test_instrument_node_still_counts_real_errors():
    @instrument_node("score_match")
    async def boom(_state):
        raise RuntimeError("boom")

    before = counter(NODE_ERRORS, node="score_match")
    with pytest.raises(RuntimeError):
        await boom({})
    assert counter(NODE_ERRORS, node="score_match") == before + 1


async def test_human_review_pause_is_not_a_node_error(install_fake_llm, dup_responses):
    install_fake_llm(dup_responses)
    before = {n: counter(NODE_ERRORS, node=n) for n in ("human_review", "supervisor")}
    graph = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = await graph.ainvoke(INPUT, config)
    assert result["__interrupt__"]
    assert (await graph.aget_state(config)).values["status"] == AnalysisStatus.awaiting_approval
    assert {n: counter(NODE_ERRORS, node=n) for n in before} == before


# ------------------------------------------------------------------ fusion v2 metric


async def test_skill_resolution_metric(install_fake_decisions):
    from jobmatch.decisions import service

    install_fake_decisions(lambda state, q: 0.01)  # "no evidence" for everything
    skills = [
        SkillAssessment(skill="Go", level="required", match="full", evidence=""),  # conflict
        SkillAssessment(skill="Rust", level="required", match="partial", evidence=""),
        SkillAssessment(skill="Zig", level="required", match="none", evidence=""),  # llm
    ]
    labels = ("llm", "resolved_partial", "conflict")
    before = {r: counter(SKILL_RESOLUTION, resolution=r) for r in labels}
    checks = await service.check_skill_evidence("CV", skills)
    assert [c.resolution for c in checks] == ["conflict", "resolved_partial", "llm"]
    assert [c.final_credit for c in checks][0] == 0.5
    after = {r: counter(SKILL_RESOLUTION, resolution=r) for r in labels}
    assert {r: after[r] - before[r] for r in labels} == {
        "llm": 1,
        "resolved_partial": 1,
        "conflict": 1,
    }


# ------------------------------------------------------------------ 2 reviewer token budget


def test_get_llm_max_tokens_override():
    assert llm.get_llm().max_tokens == get_settings().llm_max_tokens
    assert llm.get_llm(max_tokens=123).max_tokens == 123
    unlimited = llm.get_llm(max_tokens=None)
    assert unlimited.max_tokens is None
    payload = unlimited._get_request_payload([HumanMessage("hi")])
    assert "max_tokens" not in payload and "max_completion_tokens" not in payload
    capped = llm.get_llm()._get_request_payload([HumanMessage("hi")])
    # langchain-openai sends the cap as max_completion_tokens (honoured by llama-server)
    assert capped["max_completion_tokens"] == get_settings().llm_max_tokens


@pytest.mark.parametrize("enabled", [True, False])
async def test_only_the_reviewer_is_uncapped(install_fake_llm, dup_responses, monkeypatch, enabled):
    monkeypatch.setattr(get_settings(), "decisions_enabled", enabled)
    # one get_llm() per structured call: the career advisor's tool-bound model is not one
    monkeypatch.setattr(get_settings(), "career_advisor_enabled", False)
    fake = install_fake_llm(dup_responses)
    graph = build_graph(checkpointer=MemorySaver())
    await graph.ainvoke(INPUT, {"configurable": {"thread_id": str(uuid.uuid4())}})
    by_schema = dict(zip((s.__name__ for s in fake.schemas), fake.llm_kwargs, strict=True))
    reviewer = "ReviewerOutput" if enabled else "Review"
    assert by_schema[reviewer]["max_tokens"] is None
    for name, kw in by_schema.items():
        if name != reviewer:
            assert kw.get("max_tokens", llm.DEFAULT) is llm.DEFAULT, name


def _length_error() -> openai.LengthFinishReasonError:
    completion = ChatCompletion.model_construct(
        id="x",
        choices=[],
        created=0,
        model="qwen",
        object="chat.completion",
        usage=CompletionUsage(completion_tokens=4096, prompt_tokens=1746, total_tokens=5842),
    )
    return openai.LengthFinishReasonError(completion=completion)


async def test_length_error_is_not_retried(monkeypatch, caplog):
    calls = []

    class Model:
        async def ainvoke(self, messages):
            calls.append(messages)
            raise _length_error()

    class Chat:
        def with_structured_output(self, *a, **kw):
            return Model()

    monkeypatch.setattr(llm, "get_llm", lambda **kw: Chat())
    with caplog.at_level("ERROR"), pytest.raises(StructuredOutputError, match="token limit"):
        await invoke_structured("review_letter", Review, [HumanMessage("x")])
    assert len(calls) == 1
    assert any("completion_tokens=4096" in r.getMessage() for r in caplog.records)
