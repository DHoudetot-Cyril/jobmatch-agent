# CLAUDE.md — jobmatch-agent

Hierarchical LangGraph multi-agent system: job offer + CV → structured analysis, match score,
skill gaps, cover letter (writer/reviewer loop) and human-in-the-loop approval.
Portfolio project: code quality, observability and evaluation matter as much as features.

## Stack
- Python 3.12, **uv** (`uv sync --all-groups`, `uv run ...`), ruff (line length 100), pytest.
- LLM: **Qwen3.8-27B via llama.cpp `llama-server`** on the HOST GPU (AMD RX 7900 XTX, ROCm),
  started with `~/start_qwen_light.sh` → OpenAI-compatible API at `http://172.17.0.1:8080/v1`,
  model alias `qwen`, 2 slots × 16k ctx, vision enabled (mmproj), Prometheus metrics at `/metrics`.
  It is NOT in Docker on purpose. Never use Ollama or a cloud LLM.
- LangGraph + langchain-openai (`ChatOpenAI` pointed at llama-server), Postgres checkpointer.
- FastAPI API, Langfuse v4 (self-hosted) tracing, Prometheus + Grafana, DeepEval evals.

## Layout
```
src/jobmatch/
  config.py            Settings (pydantic-settings). Read config ONLY via get_settings().
  llm.py               get_llm(), image_message()  — the only way to get a chat model.
  schemas.py           Pydantic domain models (JobOffer, CandidateProfile, MatchResult, ...).
  state.py             JobMatchState + graph public interface (READ THE DOCSTRING).
  documents.py         load_document(bytes, filename) -> text (txt/md/pdf, scanned pdf & images via vision)
  prompts/             Prompt templates (one module per team).
  graph/build.py       build_graph(checkpointer) -> compiled graph
  graph/supervisor.py  top-level supervisor routing between teams
  graph/teams/         analysis.py, matching.py, writing.py  (each a subgraph)
  observability/       metrics.py (Prometheus, contract), tracing.py (Langfuse)
  api/                 FastAPI app (main.py, routes.py, service.py)
tests/                 unit tests, no LLM needed (use fake chat models); `@pytest.mark.llm` = needs llama-server
evals/                 DeepEval suite + dataset (fictional data only)
data/samples/          fictional offers & CVs; data/private/ is git-ignored (real personal data)
infra/                 prometheus/, grafana/ (provisioning + dashboards)
docker-compose.yml     full stack
```

## Contracts (do not break)
- Node names = Prometheus `node` label = Langfuse span names:
  `supervisor, parse_offer, parse_cv, score_match, analyze_gaps, write_letter, review_letter, human_review`.
- Metric names are defined in `observability/metrics.py`; the Grafana dashboard depends on them.
- Wrap every node with `@instrument_node(name)`; call `record_llm_usage(name, msg)` after LLM calls.
- Structured output: `get_llm().with_structured_output(Model, method="json_schema", include_raw=True)`.

## Ports (all bound to 127.0.0.1)
| Service | Host port |
|---|---|
| api (FastAPI) | 8000 |
| langfuse-web | 3000 |
| grafana | 3001 |
| prometheus | 9090 |
| app-db (Postgres, LangGraph checkpoints) | 5433 |
| minio (Langfuse blob storage API) | 9190 |
| llama-server (host, not Docker) | 172.17.0.1:8080 |

Inside compose, containers reach the LLM at `http://172.17.0.1:8080/v1` (tested OK).

## Commands
- `make up` / `make down` / `make logs` — stack
- `make test` — unit tests (no LLM) · `make lint` — ruff
- `make eval` — DeepEval suite (needs llama-server running)

## Rules
- English for code, comments, commit messages. README is bilingual EN/FR.
- Never commit `.env` or anything in `data/private/`. Sample data must be fictional.
- Pin Docker image versions (no `latest`).
