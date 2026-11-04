# Operations guide

How to set up, run, upgrade and troubleshoot JobMatch on a new machine. Everything runs
locally: the LLM on the host GPU, the rest in Docker Compose.

- [1. Hardware and software](#1-hardware-and-software)
- [2. The LLM server (llama.cpp)](#2-the-llm-server-llamacpp)
- [3. Configuration (environment variables)](#3-configuration-environment-variables)
- [4. Ports](#4-ports)
- [5. Make targets](#5-make-targets)
- [6. First start](#6-first-start)
- [7. Upgrading pinned versions](#7-upgrading-pinned-versions)
- [8. Data and privacy](#8-data-and-privacy)
- [9. Backup and clean-up](#9-backup-and-clean-up)
- [10. Troubleshooting](#10-troubleshooting)

```mermaid
flowchart LR
    subgraph Host
        LLM["llama-server :8080<br/>bound to 172.17.0.1"]
    end
    subgraph Compose["Docker Compose (ports on 127.0.0.1)"]
        API["api :8000"]
        JEV["jevos :8017"]
        DB[("app-db :5433")]
        LF["Langfuse :3000<br/>+ worker, ClickHouse,<br/>Postgres, Redis, MinIO"]
        PROM["Prometheus :9090"]
        GRAF["Grafana :3001"]
    end
    API -->|"http://172.17.0.1:8080/v1"| LLM
    API -->|"http://jevos:8017"| JEV
    API --> DB
    API -->|traces| LF
    PROM -->|scrape| API
    PROM -->|scrape| LLM
    GRAF --> PROM
```

## 1. Hardware and software

| Item | Used in this project | Notes |
|---|---|---|
| GPU | AMD Radeon RX 7900 XTX, 24 GB VRAM (gfx1100), ROCm / HIP | Any GPU supported by a llama.cpp backend works (CUDA, Vulkan, Metal, ROCm). The app only needs an OpenAI-compatible endpoint. |
| CPU / RAM | AMD Ryzen 7 9800X3D (8 cores / 16 threads), 30 GB RAM | Source: `bench/decisions/BACKENDS.md`. jevos runs on this CPU. |
| OS | Linux | |
| LLM | `Qwen3.8-27B-UD-IQ4_XS.gguf` (13.6 GB file) + `mmproj-BF16.gguf` (0.9 GB, vision) | Served by llama.cpp `llama-server`, alias `qwen`. |
| Containers | Docker with the Compose plugin | All images pinned (see [section 7](#7-upgrading-pinned-versions)). |
| Python tooling | [uv](https://docs.astral.sh/uv/), Python 3.12 | Only needed on the host for tests, evals and the benchmark. |

The LLM is **not** in Docker on purpose: it needs direct GPU access, and keeping it on the host
lets several projects share one loaded model.

## 2. The LLM server (llama.cpp)

### Build

llama.cpp was built with the HIP (ROCm) backend for the RX 7900 XTX. The relevant CMake cache
values on the dev machine are `GGML_HIP=ON`, `AMDGPU_TARGETS=gfx1100`,
`CMAKE_BUILD_TYPE=Release`:

```bash
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp && cd ~/llama.cpp
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

For another GPU, use the matching llama.cpp backend (for example `-DGGML_CUDA=ON` or
`-DGGML_VULKAN=ON`) and drop the two ROCm environment variables from the launcher.

Put the model files in `~/llama.cpp/models/` (the launcher expects
`Qwen3.8-27B-UD-IQ4_XS.gguf` and `mmproj-BF16.gguf`).

### Start: `~/start_qwen_light.sh`

```bash
~/start_qwen_light.sh                    # defaults: 172.17.0.1:8080, 32k ctx, 2 slots, vision on
VISION=0 ~/start_qwen_light.sh           # text only
HOST=127.0.0.1 ~/start_qwen_light.sh     # host-only (containers can't reach it)
make qwen                                # prints the command and a check
curl -s http://172.17.0.1:8080/v1/models # check
```

Flags, one by one:

| Flag / variable | Value | Why |
|---|---|---|
| `HSA_OVERRIDE_GFX_VERSION=11.0.0` | env | ROCm: makes the runtime treat the GPU as gfx 11.0.0 (RX 7900 XTX family). |
| `HIP_VISIBLE_DEVICES=0` | env | ROCm: use only the first GPU. |
| `-m …/Qwen3.8-27B-UD-IQ4_XS.gguf` | | The model weights (IQ4_XS quantization, about 4 bits per weight). |
| `--mmproj …/mmproj-BF16.gguf` | if `VISION=1` (default) | Vision projector: lets Qwen read scanned PDFs and images (`documents.py`). |
| `--alias qwen` | | Model name returned by `/v1/models`; matches `LLM_MODEL=qwen`. |
| `-c "$CTX"` | 32768 | Total context. Shared by the slots, so each slot gets 16k tokens. |
| `-np "$SLOTS"` | 2 | Two parallel slots: the agent graph and the DeepEval judge (or two eval cases) run at the same time. |
| `-ngl 99` | | Offload all layers to the GPU. |
| `-fa on` | | Flash attention: less memory and faster attention. |
| `--cache-type-k q8_0`, `--cache-type-v q8_0` | | KV cache quantized to 8 bits: about half the memory of f16 with a negligible quality loss. |
| `--jinja` | | Use the chat template embedded in the GGUF (Qwen's own Jinja template) instead of a built-in approximation. It also makes `chat_template_kwargs` work, such as `enable_thinking: false` used by the `qwen-logits` decision backend. |
| `--reasoning-budget "${REASONING_BUDGET:-1024}"` | 1024 | Caps Qwen's "thinking" tokens per answer, so one call cannot spend minutes reasoning. |
| `--metrics` | | Exposes Prometheus metrics (`llamacpp:*`) at `/metrics`, scraped by Prometheus. |
| `--host "$HOST"` | 172.17.0.1 | The `docker0` bridge address (see below). |
| `--port "$PORT"` | 8080 | |

**Why bind to `172.17.0.1`?** It is the host's address on the default Docker bridge
(`docker0`). The same URL works from the host *and* from containers, and the server is not
reachable from the local network. `127.0.0.1` would be invisible to containers; `0.0.0.0` would
expose the LLM to the LAN. The address exists only while Docker is running; check it with
`ip -4 addr show docker0`.

### VRAM

Measured on the RX 7900 XTX during this project:

| Launcher | Context | KV cache | Vision | VRAM used |
|---|---|---|---|---|
| `~/start_qwen.sh` (general-purpose launcher, not used by this project) | 262,144 | q4_0 | no | 22.3 GB |
| `~/start_qwen_light.sh` with `VISION=0` | 32,768 | q8_0 | no | ~16 GB |
| `~/start_qwen_light.sh` (default) | 32,768 | q8_0 | yes | ~17.3 GB |

The light launcher leaves several GB free on a 24 GB card. jevos runs on the CPU and uses no
VRAM.

## 3. Configuration (environment variables)

`cp .env.example .env`. The defaults work as is for local development. Settings are read
**only** through `jobmatch.config.get_settings()` (pydantic-settings, `.env` file, unknown
variables ignored). Names are case-insensitive.

### Application settings (`src/jobmatch/config.py`)

| Variable | Default | In `.env.example` | Meaning |
|---|---|---|---|
| `LLM_BASE_URL` | `http://172.17.0.1:8080/v1` | yes | OpenAI-compatible endpoint of llama-server. |
| `LLM_MODEL` | `qwen` | yes | Model name (the llama-server `--alias`). |
| `LLM_API_KEY` | `not-needed` | no | Sent to the endpoint; llama-server ignores it. |
| `LLM_TEMPERATURE` | `0.2` | no | Sampling temperature of agent calls. |
| `LLM_MAX_TOKENS` | `4096` | no | Max completion tokens per call, thinking tokens included. The reviewer (`review_letter`) ignores it and runs uncapped (`REVIEWER_MAX_TOKENS = None` in `graph/teams/writing.py`), bounded only by the 16k per-slot context. |
| `LLM_TIMEOUT_S` | `300.0` | no | Per-call HTTP timeout (seconds). |
| `MAX_REVIEW_ITERATIONS` | `2` | no | Number of reviews per writing round (writer/reviewer loop cap). |
| `MIN_TEXT_CHARS_FOR_PDF` | `200` | no | Below this many extracted characters a PDF is treated as scanned and read with vision. |
| `DECISIONS_ENABLED` | `true` | yes | Decision layer on/off. `false` = the historical LLM-only pipeline (A/B switch). |
| `DECISIONS_BACKEND` | `jevos` | yes | Backend name from `decisions/registry.py`. Only `jevos` is benchmarked and calibrated; any other backend runs with raw probabilities (T = 1.0) and a one-time warning (`calibration_temperature` in `decisions/service.py`). |
| `JEVOS_URL` | `http://127.0.0.1:8017` | yes | jevos server. Compose sets `http://jevos:8017` for the api container. |
| `DECISIONS_TIMEOUT_S` | `30.0` | yes | Timeout of one decision HTTP call. |
| `DECISIONS_CONCURRENCY` | `4` | yes | Max in-flight decisions per event loop. |
| `CLAIM_DISMISS_THRESHOLD` | `0.8` | yes | A reviewer-flagged claim is dismissed when calibrated P(supported) ≥ this. |
| `DECISION_CONFLICT_LOW` | `0.2` | yes | Fusion v2: an LLM `full` rating becomes partial credit 0.5 (resolution `conflict`) when P(evidence) < this. |
| `DECISION_CONFLICT_HIGH` | `0.8` | yes | Fusion v2: an LLM `none` rating becomes partial credit 0.5 (resolution `conflict`) when P(evidence) > this. |
| `DATABASE_URL` | `postgresql://jobmatch:jobmatch@localhost:5433/jobmatch` | yes | LangGraph checkpointer (application Postgres). Compose overrides it for the api container. |
| `LANGFUSE_ENABLED` | `true` | yes | Tracing on/off. |
| `LANGFUSE_HOST` | `http://localhost:3000` | yes | Langfuse URL. Compose overrides it with `http://langfuse-web:3000`. |
| `LANGFUSE_PUBLIC_KEY` | `pk-lf-jobmatch-local` | yes | Project API key (provisioned at first start). |
| `LANGFUSE_SECRET_KEY` | `sk-lf-jobmatch-local` | yes | Project API secret. |
| `MAX_UPLOAD_MB` | `10` | no | Max size of an uploaded file. |

Other variables read by the code:

| Variable | Default | Meaning |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Root log level of the API (`api/main.py`). |
| `JEVOS_BIN` | unset | Path of a `jev` binary the jevos backend may spawn when nothing answers at `JEVOS_URL` (`decisions/jevos_backend.py`). |

### Stack variables (`.env.example`, used by `docker-compose.yml`)

| Variable | Default | Meaning |
|---|---|---|
| `APP_DB_USER` / `APP_DB_PASSWORD` / `APP_DB_NAME` | `jobmatch` / `jobmatch` / `jobmatch` | Application Postgres (`app-db`). |
| `LANGFUSE_INIT_ORG_ID` / `LANGFUSE_INIT_ORG_NAME` | `jobmatch-org` / `JobMatch` | Organisation created by Langfuse headless init. |
| `LANGFUSE_INIT_PROJECT_ID` / `LANGFUSE_INIT_PROJECT_NAME` | `jobmatch` / `jobmatch-agent` | Project created at init (the project id appears in Langfuse URLs). |
| `LANGFUSE_INIT_PROJECT_PUBLIC_KEY` / `LANGFUSE_INIT_PROJECT_SECRET_KEY` | `pk-lf-jobmatch-local` / `sk-lf-jobmatch-local` | Project API keys created at init. Must equal `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`. |
| `LANGFUSE_INIT_USER_EMAIL` / `LANGFUSE_INIT_USER_NAME` / `LANGFUSE_INIT_USER_PASSWORD` | `admin@jobmatch.local` / `admin` / `jobmatch-admin` | Langfuse UI login created at init. |
| `LANGFUSE_NEXTAUTH_SECRET` | `change-me-nextauth-secret` | Session signing secret of the Langfuse UI. |
| `LANGFUSE_SALT` | `change-me-salt` | Salt for API key hashing. |
| `LANGFUSE_ENCRYPTION_KEY` | 64 zeros | 256-bit key (hex) for encrypting sensitive data. Generate with `openssl rand -hex 32`. |
| `LANGFUSE_DB_PASSWORD` | `langfuse` | Langfuse's own Postgres. |
| `CLICKHOUSE_PASSWORD` | `clickhouse` | Langfuse's ClickHouse (user `clickhouse`). |
| `REDIS_PASSWORD` | `redis` | Langfuse's Redis. |
| `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` | `minio` / `miniosecret` | Langfuse blob storage (S3 API). |
| `GRAFANA_ADMIN_PASSWORD` | `admin` | Grafana `admin` user password. |

### Launcher variables (`~/start_qwen_light.sh`)

| Variable | Default | Meaning |
|---|---|---|
| `HOST` | `172.17.0.1` | Bind address. |
| `PORT` | `8080` | Port. |
| `CTX` | `32768` | Total context (shared by slots). |
| `SLOTS` | `2` | Parallel slots. |
| `VISION` | `1` | `0` drops `--mmproj`. |
| `REASONING_BUDGET` | `1024` | Thinking-token cap. |

## 4. Ports

All published ports are bound to `127.0.0.1` (nothing is reachable from the network).

| Service | Host address | Purpose |
|---|---|---|
| api (FastAPI) | 127.0.0.1:8000 | `/docs`, `/analyses`, `/health`, `/ready`, `/metrics` |
| langfuse-web | 127.0.0.1:3000 | Langfuse UI and public API |
| grafana | 127.0.0.1:3001 | Dashboards |
| prometheus | 127.0.0.1:9090 | Metrics and query UI |
| app-db (Postgres) | 127.0.0.1:5433 | LangGraph checkpoints |
| minio | 127.0.0.1:9190 | Langfuse blob storage (S3 API) |
| jevos | 127.0.0.1:8017 | Decision model (`jev serve`) |
| llama-server (host) | 172.17.0.1:8080 | LLM, not in Docker |

langfuse-worker, langfuse-postgres, clickhouse and redis publish no port.

## 5. Make targets

| Target | What it does |
|---|---|
| `make` / `make help` | List targets. |
| `make up` | Build the api image and start the full stack (`docker compose up -d --build`). |
| `make down` | Stop the stack, keep volumes. |
| `make logs` | Follow logs; `make logs S=api` for one service. |
| `make ps` | Service status. |
| `make infra` | Start everything except the api (for `uvicorn --reload` on the host). |
| `make test` | Unit tests, no LLM (`pytest -m "not llm"`). |
| `make lint` | `ruff check` + `ruff format --check`. |
| `make format` | Auto-fix lint issues and format. |
| `make eval` | DeepEval suite (needs llama-server). Options via `ARGS="..."`, see `evals/README.md`. |
| `make eval-compare` | Compare two eval result directories (`A=evals/baseline B=evals/results/jevos`), no LLM. |
| `make eval-stability` | N graph runs of a few cases with decisions off then on, then compare. |
| `make qwen` | Print the llama-server start command and a check. |
| `make clean` | Stop the stack **and delete all volumes** (asks for confirmation). |

## 6. First start

```bash
git clone <repo> && cd jobmatch-agent
cp .env.example .env
uv sync --all-groups            # host tooling (tests, evals, bench)
~/start_qwen_light.sh           # separate terminal
make up
make ps                         # wait until every service is "healthy"
curl -s 127.0.0.1:8000/ready    # {"ready": true, "degraded": false, ...} = database + LLM + jevos OK
```

What happens on the first `make up`:

- **Images.** The api image is built from the `Dockerfile` (uv, `--no-dev`, non-root user).
  The jevos image is built from `infra/jevos/Dockerfile`: it downloads the pinned `jevos-v5`
  release (binary + OpenVINO INT8 model, about 700 MB) and checks both SHA-256 sums.
- **Langfuse headless init.** `langfuse-web` creates the organisation, project, API keys and
  admin user from the `LANGFUSE_INIT_*` variables, so the api can send traces with no manual
  step. Log in at http://127.0.0.1:3000 with `admin@jobmatch.local` / `jobmatch-admin`.
  ClickHouse migrations run at first boot; the healthcheck allows a 90 s start period.
- **Grafana provisioning.** The Prometheus datasource (`uid: prometheus`) and the
  `JobMatch — Agents & LLM` dashboard (folder `JobMatch`, uid `jobmatch-overview`) are loaded
  from `infra/grafana/`. Open http://127.0.0.1:3001/d/jobmatch-overview (`admin` / `admin`).
  The dashboard file is re-read every 30 s; UI edits are allowed but are not written back to
  the JSON file.
- **jevos cold start.** `jev serve` loads and warms the model before `/health` reports
  `"status": "ready"` (about 20-30 s according to `infra/jevos/Dockerfile`; the healthcheck
  allows 90 s). Check with `curl -s 127.0.0.1:8017/health`.
- **Start order.** The api waits for `app-db` and `langfuse-web` to be healthy. `jevos` is
  a soft dependency (`condition: service_started`): it is started first, but the api does not
  wait for it to be healthy and works without it. Until jevos is ready, analyses run LLM-only
  and `/ready` reports `"degraded": true`. Grafana waits for Prometheus. The Langfuse services wait for their Postgres, ClickHouse,
  Redis and MinIO.
- **Prometheus targets.** http://127.0.0.1:9090/targets should show `jobmatch-api`,
  `llama-server` and `prometheus` as `up`.

Note: `LANGFUSE_INIT_*` and `GRAFANA_ADMIN_PASSWORD` are applied when the data volumes are
created. Changing them later has no effect until the volumes are recreated (or the value is
changed in the UI).

## 7. Upgrading pinned versions

No image uses `latest`. Current pins:

| Component | Pin | Where |
|---|---|---|
| Python base (api, jevos) | `python:3.12.13-slim-trixie` | `Dockerfile`, `infra/jevos/Dockerfile` |
| uv | `ghcr.io/astral-sh/uv:0.13.7` | `Dockerfile` |
| jevos | release `jevos-v5` + `JEV_TGZ_SHA256`, `JEVOS_MODEL_SHA256` | `infra/jevos/Dockerfile` |
| Postgres (app-db, langfuse-postgres) | `postgres:17.11` | `docker-compose.yml` |
| Prometheus | `prom/prometheus:v3.15.1` | `docker-compose.yml` |
| Grafana | `grafana/grafana:13.3.0` | `docker-compose.yml` |
| Langfuse web / worker | `langfuse/langfuse:4.75.0`, `langfuse/langfuse-worker:4.75.0` | `docker-compose.yml` |
| ClickHouse | `clickhouse/clickhouse-server:26.3.62.3` | `docker-compose.yml` |
| MinIO | `cgr.dev/chainguard/minio@sha256:3c60a2a8…` (RELEASE.2026-10-23T20-06-51Z) | `docker-compose.yml` |
| Redis | `redis:7.4.12` | `docker-compose.yml` |
| Python dependencies | `uv.lock` | `uv sync --frozen` in the Dockerfile and CI |

Procedure:

1. Read the release notes (breaking changes, migrations).
2. Change the tag. Keep `langfuse` and `langfuse-worker` on the **same version**, and compare
   the Langfuse services with the official compose file of the new version (ours is adapted
   from v4.75.0).
3. `docker compose pull && make up`, then check `make ps`, `/ready`, the Prometheus targets and
   one analysis.
4. **jevos:** change `JEVOS_RELEASE` and both SHA-256 values (copy them from the release's
   `SHA256SUMS.txt`). A new model invalidates the benchmark and the calibration temperatures
   in `decisions/service.py`: re-run `bench/decisions/run_bench.py --backends jevos,jevos@q-en`
   and update `TEMPERATURE` from `results/summary.json`.
5. **Postgres major versions** need a dump/restore (see below), not just a tag change.

## 8. Data and privacy

- Everything runs on this machine: the LLM, the decision model, the judge and the tracing.
  No document is sent to a cloud service. DeepEval telemetry is disabled (`evals/__init__.py`).
- Sample data (`data/samples/`, `evals/dataset/`, `bench/decisions/dataset/`) is fictional.
- Put real CVs and offers in `data/private/`, which is git-ignored. `.env` and
  `evals/results/` are git-ignored too.
- Uploaded documents and generated letters end up in the app-db checkpoints and in Langfuse
  traces (inputs and outputs of every LLM call). Treat those volumes as personal data.
- **All credentials in `.env.example` are local-only defaults.** Before exposing any port
  beyond `127.0.0.1`, change every password and secret (Langfuse secrets with
  `openssl rand -hex 32`), and put an authenticating reverse proxy in front of the API: the
  API itself has no authentication.

## 9. Backup and clean-up

Named volumes: `app_db_data`, `prometheus_data` (15-day retention), `grafana_data`,
`langfuse_postgres_data`, `langfuse_clickhouse_data`, `langfuse_clickhouse_logs`,
`langfuse_minio_data`, `langfuse_redis_data`.

```bash
# application checkpoints (analyses)
docker compose exec -T app-db pg_dump -U jobmatch jobmatch > backup_app_db.sql
# restore into a fresh volume
docker compose exec -T app-db psql -U jobmatch jobmatch < backup_app_db.sql
```

For a full copy (including Langfuse), stop the stack (`make down`) and archive the volumes.

| Command | Effect |
|---|---|
| `make down` | Stops containers, **keeps** all data. |
| `make clean` | `docker compose down -v`: deletes **all** volumes (analyses, Langfuse traces and scores, Grafana state, Prometheus history). Asks for confirmation. |
| `rm -rf evals/results/<label>` | Deletes cached eval outputs for one label. |

## 10. Troubleshooting

Real issues hit while building this project.

### Langfuse: `GET /api/public/traces` returns an error

Langfuse v4 runs in `events_only` mode; the legacy read endpoint answers:
*"This endpoint is not available on deployments running in Langfuse v4 events_only mode…"*.
Traces are written through OpenTelemetry (the SDK does this) and scores can still be posted to
`POST /api/public/scores`. To read traces, use the UI, or query ClickHouse directly
(spans are in `events_core`; the legacy `traces` table stays empty):

```bash
docker compose exec -T clickhouse clickhouse-client --user clickhouse --password clickhouse -q "
  SELECT trace_id, any(session_id) AS session, min(start_time) AS started, count() AS spans
  FROM events_core WHERE trace_name = 'jobmatch-analysis'
  GROUP BY trace_id ORDER BY started DESC LIMIT 5 FORMAT PrettyCompact"
```

### `host.docker.internal` does not resolve

It does not exist on the Linux host itself (and needs `extra_hosts` in containers). Use
`172.17.0.1`, the `docker0` bridge address, which works from both sides. That is why
llama-server binds to it and why `LLM_BASE_URL` and the Prometheus target use it.

### LangGraph warning: msgpack "unregistered type"

When a checkpoint is loaded (resume after the human decision), LangGraph warns about
deserializing types that are not registered, and future versions will refuse them. Fixed by
`checkpoint_serializer()` in `graph/build.py`, which registers every class of
`jobmatch.schemas` with `JsonPlusSerializer(allowed_msgpack_modules=...)`. The API passes it to
`AsyncPostgresSaver(..., serde=checkpoint_serializer())`. Use it for any new checkpointer.

### Hugging Face 401 on public models (Laya backend)

An expired token in `~/.cache/huggingface/token` makes downloads of public models fail. Run with
`HF_HUB_DISABLE_IMPLICIT_TOKEN=1` or refresh the token (`bench/decisions/BACKENDS.md`).

### GitHub refuses to push `.github/workflows/ci.yml`

A push that adds or changes a workflow file is refused when the token lacks the `workflow`
scope. Fix: `gh auth refresh -s workflow`, then push again.

### jevos: HTTP 422 / decisions `skipped`

jevos accepts at most 8192 tokens of state and never truncates. `decisions/service.py` does
not send texts above 24,000 characters (a conservative ~3 chars/token bound) and treats an
HTTP 422 as "skipped": the graph keeps the LLM-only result and
`jobmatch_decisions_total{outcome="skipped"}` increases. Truncating was rejected on purpose: a
cut CV would produce false "no evidence" answers.

### The api container does not start

The api waits for `app-db` and `langfuse-web` to be **healthy**. On the first start Langfuse
needs up to 90 s (ClickHouse migrations). Check `make ps`, then `make logs S=langfuse-web` or
`make logs S=app-db`. jevos does not block the api start (soft dependency).

### `/ready` says `"degraded": true`

The `decisions` check (jevos `/health`) failed. It is non-critical: `ready` stays `true` and
analyses run with LLM-only results; `jobmatch_decisions_total{outcome="error"}` counts the
fallbacks. Check `curl -s 127.0.0.1:8017/health` and `make logs S=jevos`; on a cold start
jevos needs 20-30 s to load and warm the model.

### Analysis failed with `LengthFinishReasonError` / "truncated by the token limit"

Qwen's thinking tokens count toward `max_tokens`. In the A/B eval of 2026-11-04 one case
(`employment_gap_accountant`, decisions on) failed this way: `completion_tokens=4096`
(`evals/results/jevos/report.md`). The reviewer now runs **without a `max_tokens` cap**
(`REVIEWER_MAX_TOKENS = None`, `graph/teams/writing.py`), so it is bounded only by the 16k
per-slot context. A truncated reply is no longer retried (the identical call would hit the same
limit): `invoke_structured` fails fast with a `StructuredOutputError` that names the node and
the token counts (`graph/structured.py`). If another node hits the limit, raise
`LLM_MAX_TOKENS`.

### `/ready` reports the LLM as down

`curl -s http://172.17.0.1:8080/v1/models` from the host. If it fails: llama-server is not
running, it was started with `HOST=127.0.0.1`, or Docker (hence `docker0`) was not up when it
started. Start Docker first, then the launcher.

### Installing Docker: `sudo` needs a real terminal

Installing Docker and adding your user to the `docker` group need `sudo` with a password
prompt. In this project, `sudo` typed through Claude Code's `!` shell prefix could not read the
password: run those commands from a regular terminal, then log out and in again for the group
change.
