# ADR 0001: llama.cpp `llama-server` on the host, not Ollama, not in Docker

- **Status:** accepted
- **Related code:** `src/jobmatch/llm.py`, `src/jobmatch/config.py`, `docker-compose.yml`,
  `infra/prometheus/prometheus.yml`

## Context

The project must run fully locally (CVs are personal data, no cloud LLM) on one consumer GPU,
an AMD RX 7900 XTX with ROCm. The agents need, from the serving layer:

- an OpenAI-compatible API, so LangChain's `ChatOpenAI` can be used unchanged;
- structured output constrained by a JSON schema (`response_format: json_schema`);
- vision, to transcribe scanned PDFs and images (`--mmproj`);
- several parallel slots (the analysis team parses the offer and the CV in parallel);
- Prometheus metrics, to monitor the LLM next to the application;
- token log-probabilities, used by the `qwen-logits` decision backend
  (`src/jobmatch/decisions/qwen_logits.py`).

## Decision

Serve Qwen3.8-27B with llama.cpp's `llama-server`, started on the **host** by
`~/start_qwen_light.sh` (2 slots, `--jinja`, `--metrics`, mmproj for vision). Model alias
`qwen`. Containers reach it at `http://172.17.0.1:8080/v1` (Docker bridge address), and
Prometheus scrapes `172.17.0.1:8080/metrics`. Ollama and cloud LLMs are not used.

The features listed above are documented as verified on the live server in the `llm.py`
docstring (tool calling with `--jinja`, `json_schema` structured output, vision through
`image_url` parts, Qwen "thinking" returned in `reasoning_content` and not in `content`).

## Consequences

- **Good:** direct control of the llama.cpp flags (context, slots, quantization, mmproj);
  native `/metrics` exposed as `llamacpp:*` in Prometheus; no GPU passthrough into containers.
- **Good:** the app depends only on an OpenAI-compatible URL (`LLM_BASE_URL`), so another
  server could be swapped in through configuration.
- **Bad:** `make up` does not start the LLM. It is a separate manual step, and the start
  script lives outside the repository (`~/start_qwen_light.sh`), which makes the setup less
  reproducible for someone else.
- **Bad:** `172.17.0.1` assumes the default Linux Docker bridge.
- **Mitigation:** `GET /ready` checks the LLM (`/v1/models`, then `/health`) and returns 503
  when it is down; `make qwen` prints the start command.
