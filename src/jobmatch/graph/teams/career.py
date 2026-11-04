"""Career team: the ONE autonomous, tool-using agent of the project (ReAct loop).

    START ─┬─> career_advisor ⇄ career_tools ─> career_plan ─> END
           ├─> career_plan    (no skill gap: agent skipped, empty plan with a summary)
           └─> END            (CAREER_ADVISOR_ENABLED=false: no-op)

Everything else in the pipeline is a deterministic workflow (ADR 0002). Here the LLM decides
which tools to call, with which arguments, in which order and when to stop:

- `search_skill_taxonomy`     local ESCO snapshot (normalise a gap, find related skills)
- `check_cv_evidence`         decision model: does the CV show <related skill>? The CV text is
                              injected from the graph state (`InjectedState`), never passed
                              through the LLM.
- `search_learning_resources` curated local catalog (the only allowed source of URLs)

The CODE bounds the agent: at most `career_max_tool_rounds` tool-calling LLM turns and a
`career_timeout_s` wall-time budget for the loop; past either limit the final answer is forced
(that last call gets max(remaining budget, MIN_PLAN_BUDGET_S), then the deterministic fallback).
`career_plan` then asks for the structured `CareerPlan` from a digest of the conversation and
post-validates it against what the tools actually returned in THIS run: unknown ESCO URIs and
resource URLs are dropped (hallucinations, metric `jobmatch_career_resources_dropped_total`),
transferable skills are kept only if `check_cv_evidence` said present (p >= 0.5). If the LLM or
a tool fails, the plan still completes (tools return notes; the plan falls back to a
deterministic one built from the tool results and the gap advice).

The tool-loop messages live in this subgraph's state only (`CareerState`); the parent state
receives `career_plan` and `trace` (output schema), so the main state stays small.
"""

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import InjectedState, ToolNode

from jobmatch import llm
from jobmatch.career import tools as career_tools
from jobmatch.config import get_settings
from jobmatch.graph.structured import invoke_structured
from jobmatch.observability.metrics import (
    CAREER_RESOURCES_DROPPED,
    CAREER_ROUNDS,
    CAREER_TOOL_CALLS,
    instrument_node,
    record_llm_usage,
)
from jobmatch.prompts.career import (
    CAREER_ADVISOR_HUMAN,
    CAREER_ADVISOR_SYSTEM,
    CAREER_PLAN_PROMPT,
)
from jobmatch.schemas import (
    CandidateProfile,
    CareerPlan,
    GapPlan,
    JobOffer,
    LearningResource,
    MatchResult,
    SkillGap,
    SkillLevel,
)

log = logging.getLogger(__name__)

EVIDENCE_THRESHOLD = 0.5  # check_cv_evidence p >= this (and present) -> transferable
# The final structured answer always gets at least this much time, even when the loop used its
# whole budget (smoke on the samples: ~85 s for 4 gaps / 20 tool results, Qwen thinking on).
MIN_PLAN_BUDGET_S = 180.0
MAX_TOOL_LIMIT = 10  # clamp of the `limit` argument chosen by the LLM
MAX_RESEARCH_CHARS = 12_000  # digest of the tool loop fed to career_plan

NO_GAP_SUMMARY = {
    "fr": "Aucun écart de compétence identifié pour cette offre : le profil couvre les "
    "compétences demandées. Mettez en avant ces points forts dans la candidature.",
    "en": "No skill gap identified for this offer: the profile covers the requested skills. "
    "Highlight these strengths in the application.",
}
FALLBACK_SUMMARY = {
    "fr": "Plan établi automatiquement à partir des recherches de l'agent (synthèse LLM "
    "indisponible) : traitez d'abord les compétences requises.",
    "en": "Plan built automatically from the advisor's research (LLM synthesis unavailable): "
    "address the required skills first.",
}


def _append(left: list[str] | None, right: list[str] | None) -> list[str]:
    return (left or []) + (right or [])


class CareerInput(TypedDict, total=False):
    """What the career team reads from the parent state (no `trace`, see `_common.TeamInput`)."""

    cv_text: str
    offer: JobOffer
    profile: CandidateProfile
    match: MatchResult


class CareerOutput(TypedDict, total=False):
    """What the career team writes back to the parent state (messages stay in the subgraph)."""

    career_plan: CareerPlan
    trace: Annotated[list[str], _append]


class CareerState(CareerInput, CareerOutput, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    rounds: int  # LLM turns that requested tools
    started_at: float  # wall clock (time.time(), survives a checkpoint reload)


# --- tools (thin LangChain wrappers around jobmatch.career.tools) ----------------------------


def _status(tool_name: str, result: dict) -> str:
    if tool_name == "check_cv_evidence":
        return "empty" if result.get("present") is None else "ok"
    return "ok" if result.get("results") else "empty"


async def _run_tool(
    name: str, fn: Callable[..., Awaitable[dict]], **kwargs: Any
) -> tuple[str, dict]:
    """Call a career tool; never raise (the LLM gets a note), count the call by status."""
    try:
        result = await fn(**kwargs)
        status = _status(name, result)
    except Exception as exc:  # a broken tool must not break the agent
        log.warning("career tool %s failed: %r", name, exc)
        result = {"results": [], "note": f"tool error ({type(exc).__name__}), try another way"}
        if name == "check_cv_evidence":
            result = {"skill": kwargs.get("skill"), "p_evidence": None, "present": None,
                      "note": result["note"]}  # fmt: skip
        status = "error"
    CAREER_TOOL_CALLS.labels(tool=name, status=status).inc()
    return json.dumps(result, ensure_ascii=False), result


def _limit(limit: int, default: int) -> int:
    try:
        return max(1, min(int(limit), MAX_TOOL_LIMIT))
    except (TypeError, ValueError):
        return default


@tool(response_format="content_and_artifact")
async def search_skill_taxonomy(query: str, language: str = "en", limit: int = 5):
    """Search the ESCO skill taxonomy (local snapshot). Returns matching concepts with their
    uri, preferred label, alternative labels, a short description and RELATED skills. Use it to
    normalise a skill gap and to find related skills the candidate may already have."""
    return await _run_tool(
        "search_skill_taxonomy",
        career_tools.search_skill_taxonomy,
        query=query,
        language=language,
        limit=_limit(limit, 5),
    )


@tool(response_format="content_and_artifact")
async def check_cv_evidence(skill: str, cv_text: Annotated[str, InjectedState("cv_text")]):
    """Ask a calibrated decision model whether the candidate's CV shows evidence of `skill`.
    Returns p_evidence (probability) and present (true/false). The CV is provided
    automatically. Use it before claiming a related skill is transferable."""
    return await _run_tool(
        "check_cv_evidence", career_tools.check_cv_evidence, skill=skill, cv_text=cv_text or ""
    )


@tool(response_format="content_and_artifact")
async def search_learning_resources(skill: str, limit: int = 3):
    """Look up the curated local catalog of learning resources for a skill (title, url,
    provider, kind, free, level, language, estimated hours). The ONLY allowed source of
    learning resources and URLs."""
    return await _run_tool(
        "search_learning_resources",
        career_tools.search_learning_resources,
        skill=skill,
        limit=_limit(limit, 3),
    )


TOOLS = [search_skill_taxonomy, check_cv_evidence, search_learning_resources]
_TOOL_NODE = ToolNode(TOOLS, name="career_tools")


# --- helpers ---------------------------------------------------------------------------------


def _gaps(state: CareerState) -> list[SkillGap]:
    match = state.get("match")
    gaps = list(match.gaps) if match else []
    # required gaps first (stable order otherwise)
    return sorted(gaps, key=lambda g: g.level != SkillLevel.required)


def _language(state: CareerState) -> str:
    offer = state.get("offer")
    return offer.language if offer else "en"


def _deadline(state: CareerState) -> float:
    return (state.get("started_at") or time.time()) + get_settings().career_timeout_s


def _format_gaps(gaps: list[SkillGap]) -> str:
    return "\n".join(f"- {g.skill} ({g.level.value}): {g.advice}" for g in gaps)


def _profile_summary(profile: CandidateProfile | None) -> str:
    if profile is None:
        return "(unknown)"
    lines = []
    if profile.headline:
        lines.append(profile.headline)
    if profile.skills:
        lines.append("Skills: " + ", ".join(profile.skills[:30]))
    for e in profile.experiences[:6]:
        months = f", {e.duration_months} months" if e.duration_months else ""
        lines.append(f"- {e.role}" + (f" at {e.company}" if e.company else "") + months)
    return "\n".join(lines) or "(empty profile)"


def _initial_messages(state: CareerState) -> list[AnyMessage]:
    offer = state.get("offer")
    return [
        SystemMessage(
            CAREER_ADVISOR_SYSTEM.format(max_rounds=get_settings().career_max_tool_rounds)
        ),
        HumanMessage(
            CAREER_ADVISOR_HUMAN.format(
                title=offer.title if offer else "(unknown)",
                seniority=(offer.seniority if offer else None) or "unspecified",
                language=_language(state),
                profile_summary=_profile_summary(state.get("profile")),
                gaps=_format_gaps(_gaps(state)),
            )
        ),
    ]


class Evidence:
    """What the tools actually returned during this run (ground truth for post-validation)."""

    def __init__(self, messages: list[AnyMessage]):
        calls = {
            c["id"]: c for m in messages if isinstance(m, AIMessage) for c in m.tool_calls or []
        }
        self.steps: list[tuple[str, dict, str]] = []  # (tool, args, content) in call order
        self.concepts: dict[str, str] = {}  # ESCO uri -> preferred label
        self.present: dict[str, bool] = {}  # casefolded skill -> verified present
        self.resources: dict[str, dict] = {}  # url -> catalog record
        self.resources_by_skill: dict[str, list[str]] = {}  # casefolded query -> urls
        self.concept_by_skill: dict[str, str] = {}  # casefolded query -> first uri
        for m in messages:
            if not isinstance(m, ToolMessage):
                continue
            call = calls.get(m.tool_call_id, {})
            args = call.get("args", {})
            self.steps.append((m.name or call.get("name", "?"), args, str(m.content)))
            data = m.artifact if isinstance(m.artifact, dict) else {}
            if m.name == "search_skill_taxonomy":
                results = [r for r in data.get("results", []) if r.get("uri")]
                for r in results:
                    self.concepts[r["uri"]] = r.get("preferred_label") or ""
                if results:
                    key = str(args.get("query", "")).strip().casefold()
                    self.concept_by_skill.setdefault(key, results[0]["uri"])
            elif m.name == "check_cv_evidence":
                p = data.get("p_evidence")
                ok = data.get("present") is True and p is not None and p >= EVIDENCE_THRESHOLD
                for name in (data.get("skill"), args.get("skill")):
                    if name:
                        key = str(name).strip().casefold()
                        self.present[key] = self.present.get(key, False) or ok
            elif m.name == "search_learning_resources":
                key = str(args.get("skill", "")).strip().casefold()
                for r in data.get("results", []):
                    if r.get("url"):
                        self.resources[r["url"]] = r
                        self.resources_by_skill.setdefault(key, []).append(r["url"])

    @property
    def tool_calls(self) -> int:
        return len(self.steps)

    def research(self) -> str:
        lines = [
            f"- {name}({json.dumps(args, ensure_ascii=False)}) -> {content}"
            for name, args, content in self.steps
        ]
        text = "\n".join(lines) or "(no tool call)"
        return text[:MAX_RESEARCH_CHARS]

    def resource(self, url: str) -> LearningResource:
        r = self.resources[url]
        return LearningResource(
            title=r.get("title") or url,
            url=url,
            provider=r.get("provider"),
            free=r.get("free"),
            est_hours=r.get("est_hours"),
        )


def _find_draft(gap: SkillGap, i: int, n: int, drafts: list[GapPlan]) -> GapPlan | None:
    key = gap.skill.strip().casefold()
    found = next((d for d in drafts if d.skill.strip().casefold() == key), None)
    found = found or next((d for d in drafts if d.skill.strip().casefold().startswith(key)), None)
    if found is None and len(drafts) == n:
        found = drafts[i]
    return found


def ground_plan(
    draft: CareerPlan | None, gaps: list[SkillGap], ev: Evidence, language: str
) -> tuple[CareerPlan, int]:
    """Post-validate the LLM plan against the tool results; returns (plan, dropped URLs).

    One `GapPlan` per gap (in gap order). With no draft (LLM failure), a deterministic plan is
    built from the tool results and the gap advice.
    """
    drafts = draft.gaps if draft else []
    dropped = 0
    plans = []
    for i, gap in enumerate(gaps):
        d = _find_draft(gap, i, len(gaps), drafts)
        key = gap.skill.strip().casefold()
        if d is None:  # no LLM entry for this gap: use what the tools found for it
            uri = ev.concept_by_skill.get(key)
            urls = ev.resources_by_skill.get(key, [])
            d = GapPlan(
                skill=gap.skill,
                esco_uri=uri,
                resources=[ev.resource(u) for u in dict.fromkeys(urls)],
                actions=[gap.advice] if gap.advice else [],
                priority="high" if gap.level == SkillLevel.required else "medium",
            )
        uri = d.esco_uri if d.esco_uri in ev.concepts else None
        if uri is None and d.esco_label:  # label given without (a valid) uri
            label = d.esco_label.strip().casefold()
            uri = next((u for u, lb in ev.concepts.items() if lb.casefold() == label), None)
        transferable = [
            s for s in dict.fromkeys(d.transferable_skills) if ev.present.get(s.strip().casefold())
        ]
        resources = []
        for r in d.resources:
            if r.url in ev.resources:
                if all(x.url != r.url for x in resources):
                    resources.append(ev.resource(r.url))
            else:
                dropped += 1
                log.info("career_plan: dropped resource not returned by a tool: %s", r.url)
        plans.append(
            GapPlan(
                skill=gap.skill,
                esco_uri=uri,
                esco_label=ev.concepts.get(uri) if uri else None,
                transferable_skills=transferable,
                resources=resources,
                actions=d.actions or ([gap.advice] if gap.advice else []),
                priority=d.priority,
            )
        )
    summary = (draft.summary if draft else "") or FALLBACK_SUMMARY.get(
        language, FALLBACK_SUMMARY["en"]
    )
    return CareerPlan(summary=summary, gaps=plans, tool_calls=ev.tool_calls), dropped


# --- nodes -----------------------------------------------------------------------------------


@instrument_node("career_advisor")
async def career_advisor(state: CareerState) -> dict:
    """One ReAct turn: the LLM (bound to the tools) decides what to do next."""
    started_at = state.get("started_at") or time.time()
    update: dict = {"started_at": started_at, "trace": ["career_advisor"]}
    history = list(state.get("messages") or [])
    new: list[AnyMessage] = [] if history else _initial_messages(state)
    remaining = started_at + get_settings().career_timeout_s - time.time()
    if remaining <= 0:
        return update  # budget exhausted: career_plan forces the answer
    model = llm.get_llm().bind_tools(TOOLS)
    try:
        ai = await asyncio.wait_for(model.ainvoke(history + new), timeout=remaining)
    except TimeoutError:
        log.warning("career_advisor: wall-time budget exhausted, forcing the final answer")
        return update | ({"messages": new} if new else {})
    except Exception as exc:  # degrade: the plan is built from what we have
        log.warning("career_advisor: LLM call failed (%r), forcing the final answer", exc)
        return update | ({"messages": new} if new else {})
    record_llm_usage("career_advisor", ai)
    update["messages"] = [*new, ai]
    if getattr(ai, "tool_calls", None):
        update["rounds"] = (state.get("rounds") or 0) + 1
    return update


@instrument_node("career_tools")
async def career_tools_node(state: CareerState, config: RunnableConfig) -> dict:
    """Run the tool calls of the last advisor turn (in parallel, via LangGraph's ToolNode)."""
    result = await _TOOL_NODE.ainvoke(state, config)
    messages = result.get("messages", []) if isinstance(result, dict) else result
    return {"messages": messages, "trace": ["career_tools"]}


@instrument_node("career_plan")
async def career_plan(state: CareerState) -> dict:
    """Final structured answer (forced), post-validated against the tool results."""
    gaps = _gaps(state)
    language = _language(state)
    if not gaps:
        summary = NO_GAP_SUMMARY.get(language, NO_GAP_SUMMARY["en"])
        return {"career_plan": CareerPlan(summary=summary), "trace": ["career_plan"]}

    messages = list(state.get("messages") or [])
    ev = Evidence(messages)
    last = messages[-1] if messages else None
    notes = (
        str(last.content)
        if isinstance(last, AIMessage) and not last.tool_calls and last.content
        else "(none: the tool budget was exhausted or the advisor failed)"
    )
    offer = state.get("offer")
    prompt = CAREER_PLAN_PROMPT.format_messages(
        language=language,
        title=offer.title if offer else "(unknown)",
        gaps=_format_gaps(gaps),
        research=ev.research(),
        notes=notes[:3000],
    )
    budget = max(_deadline(state) - time.time(), MIN_PLAN_BUDGET_S)
    draft: CareerPlan | None
    try:
        draft = await asyncio.wait_for(
            invoke_structured("career_plan", CareerPlan, prompt), timeout=budget
        )
    except Exception as exc:  # TimeoutError, StructuredOutputError, LLM down...
        log.warning("career_plan: structured answer failed (%r), deterministic fallback", exc)
        draft = None
    plan, dropped = ground_plan(draft, gaps, ev, language)
    CAREER_ROUNDS.observe(state.get("rounds") or 0)
    if dropped:
        CAREER_RESOURCES_DROPPED.inc(dropped)
    return {"career_plan": plan, "trace": ["career_plan"]}


# --- routing ---------------------------------------------------------------------------------


def _expired(state: CareerState) -> bool:
    return time.time() >= _deadline(state)


def route_start(state: CareerState) -> Literal["career_advisor", "career_plan", "__end__"]:
    if not get_settings().career_advisor_enabled:
        return END
    return "career_advisor" if _gaps(state) else "career_plan"


def route_after_advisor(state: CareerState) -> Literal["career_tools", "career_plan"]:
    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    wants_tools = isinstance(last, AIMessage) and bool(last.tool_calls)
    within = (state.get("rounds") or 0) <= get_settings().career_max_tool_rounds
    return "career_tools" if wants_tools and within and not _expired(state) else "career_plan"


def route_after_tools(state: CareerState) -> Literal["career_advisor", "career_plan"]:
    if (state.get("rounds") or 0) >= get_settings().career_max_tool_rounds or _expired(state):
        return "career_plan"  # bound reached: force the final answer
    return "career_advisor"


def build_career_team() -> CompiledStateGraph:
    g = StateGraph(CareerState, input_schema=CareerInput, output_schema=CareerOutput)
    g.add_node("career_advisor", career_advisor)
    g.add_node("career_tools", career_tools_node)
    g.add_node("career_plan", career_plan)
    g.add_conditional_edges(START, route_start)
    g.add_conditional_edges("career_advisor", route_after_advisor)
    g.add_conditional_edges("career_tools", route_after_tools)
    g.add_edge("career_plan", END)
    return g.compile(name="career_team")
