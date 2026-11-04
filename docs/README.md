# Documentation

Technical documentation of JobMatch Agent. Start with the project [README](../README.md) for
the overview and quick start.

| Document | Content |
|---|---|
| [architecture.md](architecture.md) | System overview, request lifecycle, agent graph, state, scoring, writer/reviewer loop, checkpointing, failure handling, module map |
| [api.md](api.md) | HTTP endpoints, request/response examples, status codes, analysis status state machine |
| [decision-layer.md](decision-layer.md) | The jevos "System 1" decision layer: `verify_skills` (fusion v2), `verify_claims`, calibration, degradation, A/B eval results |
| [observability.md](observability.md) | Langfuse traces, Prometheus metrics, Grafana dashboard |
| [operations.md](operations.md) | Running, configuring and troubleshooting the stack |
| [adr/](adr/README.md) | Architecture Decision Records (0001-0008) |

Related documentation outside `docs/`:

| Document | Content |
|---|---|
| [evals/README.md](../evals/README.md) | DeepEval suite, dataset, metrics, A/B and stability runs |
| [bench/decisions/README.md](../bench/decisions/README.md) | Decision-model benchmark method and metrics |
| [bench/decisions/results/report.md](../bench/decisions/results/report.md) | Latest benchmark results |
| [bench/decisions/BACKENDS.md](../bench/decisions/BACKENDS.md) | jevos and Laya backends: install, options, smoke results |
