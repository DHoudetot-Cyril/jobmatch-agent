"""Career advisor (the autonomous tool-using agent): scripted tool-calling LLM + fake tools."""

import uuid
from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver

from jobmatch.config import get_settings
from jobmatch.graph.build import build_graph
from jobmatch.graph.supervisor import next_step
from jobmatch.graph.teams.career import build_career_team
from jobmatch.graph.teams.matching import GapAnalysis, ScoreAssessment, SkillAssessment
from jobmatch.observability.metrics import CAREER_RESOURCES_DROPPED, CAREER_TOOL_CALLS
from jobmatch.schemas import (
    CandidateProfile,
    CareerPlan,
    CoverLetter,
    GapPlan,
    JobOffer,
    LearningResource,
    MatchResult,
    Review,
    SkillGap,
    SkillLevel,
)

CV = "CV de Camille: Python, Spark, RabbitMQ, SQL"
KAFKA_URI = "http://data.europa.eu/esco/skill/kafka"
KAFKA_URL = "https://kafka.apache.org/quickstart"


def call(name: str, i: int = 0, **args: Any) -> dict:
    return {"name": name, "args": args, "id": f"{name}-{i}-{uuid.uuid4().hex[:6]}"}


def turn(*calls: dict, content: str = "") -> AIMessage:
    return AIMessage(
        content=content,
        tool_calls=list(calls),
        usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
    )


class FakeToolLLM:
    """Scripted chat model: `bind_tools(...).ainvoke` returns the next scripted AIMessage
    (the last repeats); structured calls return the scripted CareerPlan (or raise)."""

    def __init__(self, turns: list[AIMessage], plan: CareerPlan | Exception | Callable):
        self.turns, self.plan = turns, plan
        self.tool_turns: list[list[BaseMessage]] = []
        self.plan_calls: list[list[BaseMessage]] = []
        self.bound: list[str] = []

    def bind_tools(self, tools: list) -> "FakeToolLLM":
        self.bound = [t.name for t in tools]
        return self

    async def ainvoke(self, messages: list[BaseMessage], *a: Any, **kw: Any) -> AIMessage:
        self.tool_turns.append(messages)
        msg = self.turns[min(len(self.tool_turns) - 1, len(self.turns) - 1)]
        n = len(self.tool_turns)  # fresh message / tool-call ids: a repeated turn is a new turn
        calls = [{**c, "id": f"{c['id']}-{n}"} for c in msg.tool_calls]
        return msg.model_copy(update={"id": None, "tool_calls": calls})

    def with_structured_output(self, schema: type, **kwargs: Any) -> "FakeToolLLM._S":
        assert schema is CareerPlan and kwargs.get("method") == "json_schema"
        return FakeToolLLM._S(self)

    class _S:
        def __init__(self, fake: "FakeToolLLM"):
            self.fake = fake

        async def ainvoke(self, messages: list[BaseMessage], *a: Any, **kw: Any) -> dict:
            self.fake.plan_calls.append(messages)
            plan = self.fake.plan
            if isinstance(plan, Exception):
                raise plan
            if callable(plan):
                plan = plan(messages)
            return {"raw": AIMessage(content="{}"), "parsed": plan, "parsing_error": None}


class FakeTools:
    """Replaces jobmatch.career.tools functions; records calls (cv_text included)."""

    def __init__(self, fail: set[str] | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.fail = fail or set()

    def _rec(self, name: str, kw: dict) -> None:
        self.calls.append((name, kw))
        if name in self.fail:
            raise RuntimeError(f"{name} broken")

    async def search_skill_taxonomy(self, query: str, language: str = "en", limit: int = 5):
        self._rec("search_skill_taxonomy", {"query": query, "language": language, "limit": limit})
        if "kafka" not in query.casefold():
            return {"results": [], "note": "no match"}
        return {
            "results": [
                {
                    "uri": KAFKA_URI,
                    "preferred_label": "Apache Kafka",
                    "alt_labels": [],
                    "description": "event streaming",
                    "related": [{"uri": "u:rabbitmq", "label": "RabbitMQ"}],
                }
            ],
            "note": "",
        }

    async def check_cv_evidence(self, skill: str, cv_text: str):
        self._rec("check_cv_evidence", {"skill": skill, "cv_text": cv_text})
        p = 0.9 if skill.casefold() in cv_text.casefold() else 0.1
        return {"skill": skill, "p_evidence": p, "present": p >= 0.5, "note": ""}

    async def search_learning_resources(self, skill: str, limit: int = 3):
        self._rec("search_learning_resources", {"skill": skill, "limit": limit})
        if "kafka" not in skill.casefold():
            return {"results": [], "note": "nothing in the catalog"}
        return {
            "results": [
                {
                    "title": "Kafka quickstart",
                    "url": KAFKA_URL,
                    "provider": "Apache",
                    "kind": "doc",
                    "free": True,
                    "level": "beginner",
                    "language": "en",
                    "est_hours": 3,
                }
            ],
            "note": "",
        }


@pytest.fixture
def fake_tools(monkeypatch: pytest.MonkeyPatch) -> Callable[..., FakeTools]:
    def install(**kw: Any) -> FakeTools:
        tools = FakeTools(**kw)
        for name in ("search_skill_taxonomy", "check_cv_evidence", "search_learning_resources"):
            monkeypatch.setattr(
                f"jobmatch.career.tools.{name}", getattr(tools, name), raising=False
            )
        return tools

    return install


@pytest.fixture
def install_tool_llm(monkeypatch: pytest.MonkeyPatch) -> Callable[..., FakeToolLLM]:
    def install(turns: list[AIMessage], plan: Any) -> FakeToolLLM:
        fake = FakeToolLLM(turns, plan)
        monkeypatch.setattr("jobmatch.llm.get_llm", lambda *a, **kw: fake)
        return fake

    return install


@pytest.fixture
def settings_override(monkeypatch: pytest.MonkeyPatch):
    def set_(**values: Any) -> None:
        s = get_settings()
        for k, v in values.items():
            monkeypatch.setattr(s, k, v)

    return set_


def career_input(offer, profile, gaps: list[SkillGap] | None = None) -> dict:
    gaps = (
        [SkillGap(skill="Kafka", level=SkillLevel.nice_to_have, advice="Suivre un tutoriel")]
        if gaps is None
        else gaps
    )
    match = MatchResult(score=80, matched_skills=["Python"], gaps=gaps, summary="ok")
    return {"cv_text": CV, "offer": offer, "profile": profile, "match": match}


def good_plan(**gap_overrides: Any) -> CareerPlan:
    gap = {
        "skill": "Kafka",
        "esco_uri": KAFKA_URI,
        "esco_label": "Apache Kafka",
        "transferable_skills": ["RabbitMQ"],
        "resources": [LearningResource(title="Kafka quickstart", url=KAFKA_URL)],
        "actions": ["Suivre le quickstart", "Construire un pipeline de streaming"],
        "priority": "medium",
    } | gap_overrides
    return CareerPlan(summary="Feuille de route Kafka.", gaps=[GapPlan(**gap)], tool_calls=99)


SCRIPT = [
    turn(call("search_skill_taxonomy", query="Kafka", language="en")),
    turn(
        call("check_cv_evidence", 1, skill="RabbitMQ"),
        call("search_learning_resources", 1, skill="Kafka"),
    ),
    turn(content="Kafka: ESCO trouvé, RabbitMQ transférable, 1 ressource."),
]


async def test_agent_calls_tools_then_produces_grounded_plan(
    offer, profile, fake_tools, install_tool_llm
):
    tools = fake_tools()
    fake = install_tool_llm(SCRIPT, good_plan())
    out = await build_career_team().ainvoke(career_input(offer, profile))

    assert set(out) == {"career_plan", "trace"}  # tool-loop messages stay in the subgraph
    assert out["trace"] == [
        "career_advisor",
        "career_tools",
        "career_advisor",
        "career_tools",
        "career_advisor",
        "career_plan",
    ]
    assert fake.bound == [
        "search_skill_taxonomy",
        "check_cv_evidence",
        "search_learning_resources",
    ]
    assert [n for n, _ in tools.calls] == [
        "search_skill_taxonomy",
        "check_cv_evidence",
        "search_learning_resources",
    ]
    # cv_text is injected by the graph, never chosen by the LLM
    assert tools.calls[1][1]["cv_text"] == CV

    plan = out["career_plan"]
    assert plan.tool_calls == 3  # set by the graph, not by the LLM
    [g] = plan.gaps
    assert (g.esco_uri, g.esco_label) == (KAFKA_URI, "Apache Kafka")
    assert g.transferable_skills == ["RabbitMQ"]
    assert [r.url for r in g.resources] == [KAFKA_URL]
    assert g.resources[0].provider == "Apache"  # catalog record, not the LLM's copy
    # the final answer was built from the research digest
    research = fake.plan_calls[0][-1].content
    assert "search_learning_resources" in research and KAFKA_URL in research


def test_cv_text_is_not_exposed_to_the_llm():
    from jobmatch.graph.teams.career import check_cv_evidence

    assert set(check_cv_evidence.tool_call_schema.model_json_schema()["properties"]) == {"skill"}


async def test_max_tool_rounds_enforced(
    offer, profile, fake_tools, install_tool_llm, settings_override
):
    settings_override(career_max_tool_rounds=2)
    fake_tools()
    endless = [turn(call("search_skill_taxonomy", query="Kafka"))]  # never stops by itself
    fake = install_tool_llm(endless, good_plan())
    out = await build_career_team().ainvoke(career_input(offer, profile))
    assert out["trace"].count("career_tools") == 2
    assert len(fake.tool_turns) == 2
    assert out["trace"][-1] == "career_plan" and len(fake.plan_calls) == 1
    assert "exhausted" in fake.plan_calls[0][-1].content  # no advisor notes: forced answer


async def test_timeout_forces_the_final_answer(
    offer, profile, fake_tools, install_tool_llm, settings_override
):
    settings_override(career_timeout_s=0.0)
    tools = fake_tools()
    fake = install_tool_llm(SCRIPT, good_plan())
    out = await build_career_team().ainvoke(career_input(offer, profile))
    assert tools.calls == [] and fake.tool_turns == []
    assert out["career_plan"].gaps[0].esco_uri is None  # nothing verified by a tool


async def test_url_not_returned_by_a_tool_is_dropped(offer, profile, fake_tools, install_tool_llm):
    fake_tools()
    invented = LearningResource(title="Kafka masterclass", url="https://invented.example/kafka")
    plan = good_plan(
        resources=[invented, LearningResource(title="x", url=KAFKA_URL)],
        esco_uri="http://data.europa.eu/esco/skill/invented",
        esco_label="Invented",
    )
    install_tool_llm(SCRIPT, plan)
    before = CAREER_RESOURCES_DROPPED._value.get()
    out = await build_career_team().ainvoke(career_input(offer, profile))
    [g] = out["career_plan"].gaps
    assert [r.url for r in g.resources] == [KAFKA_URL]
    assert (g.esco_uri, g.esco_label) == (None, None)
    assert CAREER_RESOURCES_DROPPED._value.get() == before + 1


async def test_transferable_skill_without_positive_evidence_is_dropped(
    offer, profile, fake_tools, install_tool_llm
):
    fake_tools()
    script = [
        turn(
            call("check_cv_evidence", 0, skill="RabbitMQ"),
            call("check_cv_evidence", 1, skill="Pulsar"),  # p = 0.1 -> absent
        ),
        turn(content="done"),
    ]
    # "Flink" was never checked, "Pulsar" was checked negative
    install_tool_llm(script, good_plan(transferable_skills=["RabbitMQ", "Pulsar", "Flink"]))
    out = await build_career_team().ainvoke(career_input(offer, profile))
    assert out["career_plan"].gaps[0].transferable_skills == ["RabbitMQ"]


async def test_no_gaps_skips_the_agent(offer, profile, fake_tools, install_tool_llm):
    tools = fake_tools()
    fake = install_tool_llm(SCRIPT, good_plan())
    out = await build_career_team().ainvoke(career_input(offer, profile, gaps=[]))
    assert out["trace"] == ["career_plan"]
    assert out["career_plan"].gaps == [] and out["career_plan"].summary
    assert fake.tool_turns == [] and fake.plan_calls == [] and tools.calls == []


async def test_disabled_is_a_noop(offer, profile, fake_tools, install_tool_llm, settings_override):
    settings_override(career_advisor_enabled=False)
    fake = install_tool_llm(SCRIPT, good_plan())
    out = await build_career_team().ainvoke(career_input(offer, profile))
    assert not (out or {}).get("career_plan") and fake.tool_turns == []
    # the supervisor skips the team entirely: previous pipeline
    state = career_input(offer, profile)
    assert next_step(state) == "writing_team"


async def test_tool_exception_is_graceful(offer, profile, fake_tools, install_tool_llm):
    fake_tools(fail={"search_skill_taxonomy", "search_learning_resources"})
    fake = install_tool_llm(SCRIPT, good_plan())
    before = CAREER_TOOL_CALLS.labels(tool="search_skill_taxonomy", status="error")._value.get()
    out = await build_career_team().ainvoke(career_input(offer, profile))
    # the LLM saw a note, not a crash
    tool_msgs = [m for m in fake.tool_turns[1] if m.type == "tool"]
    assert "tool error" in tool_msgs[0].content
    after = CAREER_TOOL_CALLS.labels(tool="search_skill_taxonomy", status="error")._value.get()
    assert after == before + 1
    [g] = out["career_plan"].gaps
    assert g.resources == [] and g.esco_uri is None  # nothing returned -> nothing kept
    assert g.transferable_skills == ["RabbitMQ"]  # the working tool still counts


async def test_plan_llm_failure_falls_back_to_tool_results(
    offer, profile, fake_tools, install_tool_llm
):
    fake_tools()
    install_tool_llm(SCRIPT, RuntimeError("llm down"))
    out = await build_career_team().ainvoke(career_input(offer, profile))
    plan = out["career_plan"]
    [g] = plan.gaps
    assert g.esco_uri == KAFKA_URI and [r.url for r in g.resources] == [KAFKA_URL]
    assert g.actions == ["Suivre un tutoriel"]  # gap advice
    assert plan.summary and plan.tool_calls == 3


async def test_full_graph_runs_career_team_before_writing(offer, profile, fake_tools, monkeypatch):
    """Top-level routing: matching -> career -> writing, plan in the interrupt payload."""
    from conftest import FakeLLM

    fake_tools()
    assessment = ScoreAssessment(
        skills=[
            SkillAssessment(skill="Python", level="required", match="full", evidence="CV"),
            SkillAssessment(skill="Apache Spark", level="required", match="full", evidence="CV"),
            SkillAssessment(skill="Kafka", level="nice_to_have", match="none", evidence="no"),
        ],
        experience_fit=80,
        summary="ok",
    )
    gap = SkillGap(skill="Kafka", level=SkillLevel.nice_to_have, advice="Tuto")
    structured = FakeLLM(
        {
            JobOffer: [offer],
            CandidateProfile: [profile],
            ScoreAssessment: [assessment],
            GapAnalysis: [GapAnalysis(gaps=[gap])],
            CareerPlan: [good_plan()],
            CoverLetter: [CoverLetter(subject="s", body="b")],
            Review: [Review(approved=True)],
        }
    )
    tool_llm = FakeToolLLM(SCRIPT, good_plan())
    structured.bind_tools = tool_llm.bind_tools  # type: ignore[attr-defined]
    monkeypatch.setattr("jobmatch.llm.get_llm", lambda *a, **kw: structured)
    graph = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = await graph.ainvoke({"offer_text": "o", "cv_text": CV}, config)

    [intr] = result["__interrupt__"]
    plan = CareerPlan.model_validate(intr.value["career_plan"])
    assert plan.gaps[0].esco_uri == KAFKA_URI and plan.tool_calls == 3
    trace = (await graph.aget_state(config)).values["trace"]
    i = trace.index("career_plan")
    assert trace.index("analyze_gaps") < trace.index("career_advisor") < i
    assert i < trace.index("write_letter")
    state = (await graph.aget_state(config)).values
    assert "messages" not in state  # tool loop kept out of the main state
