# Architecture Decision Records

Short records of the main design choices: context, decision, consequences.

| # | Decision | Status |
|---|---|---|
| [0001](0001-llama-cpp-on-host.md) | llama.cpp `llama-server` on the host GPU (ROCm), not Ollama, not in Docker | accepted |
| [0002](0002-deterministic-supervisor.md) | Deterministic supervisor routing instead of LLM routing | accepted |
| [0003](0003-score-computed-in-code.md) | Match score computed in code from per-skill LLM ratings | accepted |
| [0004](0004-hitl-interrupt-postgres.md) | Human-in-the-loop with LangGraph `interrupt` and a Postgres checkpointer | accepted |
| [0005](0005-langfuse-and-prometheus.md) | Self-hosted Langfuse v4 for traces, Prometheus + Grafana for metrics | accepted |
| [0006](0006-decision-layer-jevos.md) | Decision layer: jevos only, yes/no only, English questions, calibrated | accepted |
| [0007](0007-local-qwen-judge.md) | Local Qwen as the DeepEval judge | accepted |

New record: copy the structure of an existing one, take the next number, and add it to this table.
