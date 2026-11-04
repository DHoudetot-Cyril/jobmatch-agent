# ADR 0009: Deterministic workflow plus one bounded autonomous agent

- **Status:** accepted
- **Related code:** `src/jobmatch/graph/teams/career.py`, `src/jobmatch/career/tools.py`,
  `src/jobmatch/prompts/career.py`, `src/jobmatch/graph/supervisor.py`, `data/career/`

## Context

Everything in the graph so far is a **workflow**: the code fixes the order of the steps, and
the LLM only fills in each step (parse, rate, advise, write, review). The supervisor is a pure
function ([ADR 0002](0002-deterministic-supervisor.md)). That fits a pipeline whose steps are
always the same.

Turning skill gaps into a career plan is different. The right next step depends on what the
previous lookup returned:

- A gap may map to an ESCO concept (AWS Glue → *cloud technologies*), map only through a
  broader query (dbt → *data warehouse*), or not exist in ESCO at all (Kafka, Kubernetes,
  Docker, Terraform: `data/career/README.md`, "Not found in ESCO"). In the last case the agent
  has to broaden the query or move on.
- The related skills worth checking against the CV come from the taxonomy result, so they are
  only known after the first call.
- Learning resources are looked up per skill, and some skills have none in the catalog.

Writing this as a fixed sequence would mean coding a decision tree for every combination of
"found / not found / found something related". An agent that picks its next tool from the
results is a better fit, as long as the code keeps it bounded.

## Decision

Keep the deterministic workflow for the fixed pipeline, and add **one** autonomous,
tool-using agent where the task is open-ended: the `career_team` subgraph, run by the
supervisor after `matching_team` and before `writing_team`.

```
START ─┬─> career_advisor ⇄ career_tools ─> career_plan ─> END
       ├─> career_plan   (no gap: empty plan with a summary)
       └─> END           (CAREER_ADVISOR_ENABLED=false)
```

- **ReAct loop.** `career_advisor` is Qwen bound to three local tools (`bind_tools`). It
  decides which tools to call, with which arguments, in which order and when to stop.
  `career_tools` runs the calls (LangGraph `ToolNode`, in parallel within a turn).
- **Tools, all local, no network at runtime:** `search_skill_taxonomy` (fuzzy EN/FR search in
  an ESCO snapshot, related skills included), `check_cv_evidence` (calibrated yes/no from the
  decision layer, [ADR 0006](0006-decision-layer-jevos.md)), `search_learning_resources`
  (curated catalog with checked URLs).
- **Guardrails, enforced by code, not by the prompt:**
  - at most `CAREER_MAX_TOOL_ROUNDS` (6) tool-calling turns and `CAREER_TIMEOUT_S` (120 s) of
    wall time for the loop; past either bound the final answer is forced;
  - the final structured call (`career_plan`) always gets at least `MIN_PLAN_BUDGET_S`
    (180 s), then falls back to a deterministic plan built from the tool results and the gap
    advice;
  - the CV text is injected from the graph state into `check_cv_evidence` (`InjectedState`),
    never passed through the LLM;
  - post-validation against what the tools returned **in this run**: ESCO URIs and resource
    URLs that no tool returned are dropped (`jobmatch_career_resources_dropped_total`), and a
    transferable skill is kept only if `check_cv_evidence` said present with p ≥ 0.5;
  - a failing tool returns a note instead of raising, a failing LLM call forces the answer:
    the team never fails the analysis.
- **Explicit subgraph, not `create_react_agent`.** The loop is three plain nodes with
  conditional edges, so each step is an instrumented node (`career_advisor`, `career_tools`,
  `career_plan`) with its own metrics and Langfuse span, and the bounds are ordinary,
  unit-tested routing functions. The tool-loop messages stay in the subgraph state
  (`CareerState`); only `career_plan` reaches the main state.
- **Switch.** `CAREER_ADVISOR_ENABLED=false` makes the supervisor skip the team: exact
  previous pipeline, `career_plan` stays `null`.

## Alternatives considered

- **`langgraph.prebuilt.create_react_agent`.** Deprecated since LangGraph 1.0 (the installed
  1.2.19 points to `langchain.agents.create_agent`, a package this project does not depend
  on). It also hides the loop in one node, which makes the per-step bounds, metrics and
  post-validation harder to own.
- **Making the whole pipeline agentic** (an LLM supervisor choosing the next team). Rejected
  for the reasons of ADR 0002: the order of the main steps never changes, so an LLM router
  would add latency, tokens and a failure mode for no new capability.
- **A fixed career step** (taxonomy, then evidence, then resources, for every gap). Simpler
  and predictable, but it cannot react to "no ESCO concept" or pick which related skills to
  check, which is the point of the step.

## Consequences

- **Good:** the project shows both approaches, each where it fits: a workflow for the fixed
  pipeline, an agent for the open-ended research.
- **Good:** in the one real run on `data/samples` (2026-11-04, a smoke run, not an eval) the
  agent adapted to the tool results: 3 tool rounds and 19 tool calls (taxonomy for each of the
  4 gaps, then broader taxonomy queries and CV checks on related skills such as PySpark, SQL,
  Docker, AWS S3, CI/CD, then catalog lookups for the 4 gaps). It mapped AWS Glue →
  *cloud technologies*, dbt → *data warehouse*, Terraform → *ICT infrastructure*, Kafka → no
  ESCO concept, with 1-3 catalog resources and 3 actions per gap, in French, and 0 URLs
  dropped.
- **Bad: latency.** That run took 491 s in total versus about 320-360 s in earlier smoke runs
  without the team; `career_team` itself took about 167 s (advisor 87 s, tools 0.5 s, final
  plan 79 s). Worst case for the team is about 120 + 180 s. The final call needs about 80 s with
  Qwen's thinking on; turning thinking off for that call should speed it up, but `get_llm()`
  does not support that yet.
- **Bad: ESCO coverage.** ESCO covers few modern tech tools (no Kafka, Kubernetes, Docker,
  Terraform, dbt, Airflow, Spark, MLflow, Linux concepts), so mappings are often broad, and
  transferable skills sometimes come out as generic ESCO labels.
- **Bad: not evaluated.** The DeepEval suite has no metric for the career plan yet; one smoke
  run is all the evidence so far.
- **Less predictable than the rest:** the number and order of tool calls vary between runs.
  The bounds cap the cost, and `jobmatch_career_rounds` / `jobmatch_career_tool_calls_total`
  show the behaviour over time.
