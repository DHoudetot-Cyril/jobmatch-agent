# ADR 0005: Self-hosted Langfuse v4 for traces, Prometheus + Grafana for metrics

- **Status:** accepted
- **Related code:** `src/jobmatch/observability/tracing.py`,
  `src/jobmatch/observability/metrics.py`, `docker-compose.yml`, `infra/prometheus/`,
  `infra/grafana/`. Details: [`../observability.md`](../observability.md)

## Context

Two different questions need answers in production:

1. *What happened in this analysis?* Which prompts were sent, what each agent returned, how
   many tokens, why the reviewer rejected a letter, what the human decided.
2. *How is the system doing over time?* Request rates, node latency, errors, token usage,
   score distribution, llama-server load, decision-model outcomes.

The data includes CVs, so it must stay on the machine.

## Decision

Use **both**, each for what it is good at:

| | Langfuse v4.75.0 (self-hosted) | Prometheus v3 + Grafana 13 |
|---|---|---|
| Unit | one trace per analysis (session id = analysis id) | time series with low-cardinality labels |
| Content | prompts, outputs, spans per node, token usage, eval scores (`eval_<metric>`) | counters and histograms (`jobmatch_*`) plus `llamacpp:*` from llama-server |
| Used for | debugging one run, reviewing outputs, attaching evaluation scores | dashboards, latency and error trends, capacity |
| Integration | LangChain `CallbackHandler`, deterministic trace id seeded by the analysis id | `instrument_node`, `record_llm_usage`, HTTP middleware, `/metrics` scrape |

Langfuse runs with all its dependencies in Compose (web, worker, ClickHouse, Postgres, Redis,
MinIO), with headless initialization (`LANGFUSE_INIT_*`) so API keys exist at first boot.
Node names are a shared contract: Prometheus `node` label = Langfuse span name.

## Consequences

- **Good:** first run and resume after the human decision land in the same trace.
- **Good:** metrics stay cheap and aggregated; per-request detail stays in traces, not in
  Prometheus labels.
- **Good:** tracing is optional at runtime: if Langfuse is disabled or unreachable, the handler
  is `None` and the analysis continues.
- **Bad:** six extra containers for Langfuse, and more RAM and disk on the dev machine.
- **Bad:** two places to look; the analysis id is the key that links them (log lines,
  Langfuse session, API id).
