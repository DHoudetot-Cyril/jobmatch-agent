# Open decision-model backends

Two local, open alternatives to TypeSafe's hosted Jev, wrapped behind the `DecisionModel`
contract (`src/jobmatch/decisions/base.py`). Both run on **CPU only** (the GPU is reserved for
Qwen / llama.cpp). Accuracy and calibration are measured by the benchmark (`run_bench.py`); the
numbers below are smoke-test observations only.

Hardware for all measurements: AMD Ryzen 7 9800X3D (8 cores / 16 threads), 30 GB RAM, Linux.

| | `laya` | `jevos` |
|---|---|---|
| Class | `jobmatch.decisions.laya_backend.LayaDecisionModel` | `jobmatch.decisions.jevos_backend.JevosDecisionModel` |
| Source | [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya), [GitHub](https://github.com/NandhaKishorM/laya) | [feder-cr/jev](https://github.com/feder-cr/jev) |
| License | Apache-2.0 (code + weights) | MIT (code + weights); ships OpenVINO/oneTBB (Apache-2.0), llama.cpp tokenizer, cpp-httplib, nlohmann/json (MIT) |
| Version | PyPI `laya==0.4.22` | release `jevos-v5` (tag commit `67c78465`, 2026-10-16), model `jevos-v5` |
| Model | encoder (mmBERT-base 322M / ModernBERT-large 421M) + decision head | small decoder LLM (MiniCPM5-based tokenizer), OpenVINO INT8, asked one yes/no prompt per option/level |
| Runtime | PyTorch 2.15.0+cpu, transformers 5.17.0 (Python, in-process) | C++ binary `jev serve` (OpenVINO 2026.4 runtime), HTTP `POST /v1/systemone` |
| Languages | 100+ (multilingual checkpoint) | English only (but see smoke results: short French items worked) |
| Context | multilingual: 1024 default, up to 8192 via `max_len`; English: trained at 512 | 8192 tokens, **no truncation** (longer state = HTTP 422) |

## Install

```bash
uv sync --all-groups          # or: uv sync --group decisions
```

The `decisions` dependency group pulls `laya` and **CPU-only torch** from the explicit
`pytorch-cpu` uv index (`https://download.pytorch.org/whl/cpu`, see `[tool.uv.index]` /
`[tool.uv.sources]` in `pyproject.toml`): no CUDA/ROCm wheels. The API image is unaffected:
the Dockerfile runs `uv sync --frozen --no-dev`, which installs no extra group (checked with
`--dry-run`: no torch / laya / transformers).

**jevos is not on PyPI** (no Python package: one native binary + weights). Manual step:

```bash
mkdir -p ~/.local/share/jevos/dl && cd ~/.local/share/jevos/dl
gh release download jevos-v5 -R feder-cr/jev \
  -p jev-linux-x64.tar.gz -p jevos-v5-openvino-int8.zip -p SHA256SUMS.txt
sha256sum -c SHA256SUMS.txt --ignore-missing
cd .. && tar -xzf dl/jev-linux-x64.tar.gz && cd jev && unzip ../dl/jevos-v5-openvino-int8.zip
# -> ~/.local/share/jevos/jev/jev  and  ~/.local/share/jevos/jev/model/  (~700 MB)
```

The backend uses an already-running `jev serve` at `base_url` (default
`http://127.0.0.1:8017`); otherwise it spawns `~/.local/share/jevos/jev/jev serve` (or
`$JEVOS_BIN`, or the `binary=` arg) and stops it in `aclose()`. Without either it raises
`RuntimeError` with these instructions. (Hugging Face `feder-cr/jev` only hosts the older
`jevos-v4`; v5 is GitHub-release only.)

## Mapping of our `Question` kinds

| ours | Laya | jevos |
|---|---|---|
| `yes_no` | `noul`, P(true) -> `{"no": 1-p, "yes": p}` | `noul`, P(yes) -> same |
| `choice` | `choice`, `criteria = {label: description}`; `probabilities` keyed by label | same wire format; <= 26 options; internally one yes/no prompt per option x phrasing |
| `score`  | `score`, `criteria = ["label: description", ...]` in order; `probabilities["0".."k-1"]` mapped back to our labels by position | same; **2-10 levels**, distinct texts (more raises `ValueError`); jevos calls its scores "early" |

Outputs are re-normalised over our labels (Laya rounds to 4 decimals).
Laya: `asyncio.to_thread` + a lock (torch already uses all cores per forward pass), lazy single
load. jevos: plain async HTTP; the server batches concurrent requests on its own scheduler thread.

## Ctor options

- `LayaDecisionModel(checkpoint="multilingual" | "english" | "typed-decisions" | <repo or dir>,
  max_len=1024 (64..8192, None = checkpoint default), head_max_len=None, revision=None,
  num_threads=None)`. Named checkpoints are pinned to the revisions laya 0.4.22 reviewed
  (`multilingual` = `convaiinnovations/laya-multilingual@e4e9ddf2`, `english` =
  `convaiinnovations/laya@55cf4c4e`, `typed-decisions` = `...laya-typed-decisions@1a793eb5`).
  A truncated state logs a warning.
- `JevosDecisionModel(base_url="http://127.0.0.1:8017", binary=None, threads=None,
  state_cache=True, timeout=120)`. `state_cache=False` passes `--state-cache 0` (only when the
  backend spawns the server): every request re-reads its text, as in jevos' own benchmarks.

## Smoke results (2026-11-04)

9 items: EN/FR short CVs with/without Docker ("Does this CV show evidence of Docker?" / "Ce CV
montre-t-il une expérience de Docker ?"), a 3-way seniority choice (EN, FR) and a 4-level DevOps
score. Script: one process per backend, warm p50 over 20 calls cycling the items.

| | load (cold, cached weights) | load (first download) | warm p50 / p90 | peak RSS | answers right (of 9) |
|---|---|---|---|---|---|
| laya multilingual | 2.5 s | 11 s | 52 / 59 ms | 2.3 GB | 8 (score "none" -> "solid") |
| laya english | — | 9.8 s | 139 / 160 ms | 2.8 GB | 6 (missed EN Docker yes 0.46; both scores) |
| laya typed-decisions | — | 9.8 s | 141 / 163 ms | 2.8 GB | 7 (both scores off), probabilities flat (0.5-0.6) |
| jevos-v5 (state cache on) | 2.0 s (server ready) | n/a (manual) | 17 / 28 ms | 1.13 GB (server) + 45 MB client | 9 |
| jevos-v5 (state cache off) | 2.0 s | | 24 / 40 ms | 1.13 GB | 9 |

Selected probabilities:

| item | laya multilingual | jevos-v5 |
|---|---|---|
| EN CV + Docker, yes_no | yes 0.78 | yes 0.98 |
| FR CV + Docker, yes_no | yes 0.91 | yes 0.97 |
| FR CV, no Docker | yes 0.01 | yes 0.01 |
| EN senior (6 y), choice | senior 0.81 | senior 0.99 |
| EN junior (1 y), choice | junior 0.98 | junior 0.99 |
| FR senior (7 y), choice | senior 0.80 | senior 0.96 |
| EN DevOps expert, score | expert 0.39 / solid 0.37 | expert 0.80 |
| EN no DevOps, score | **solid 0.69** (wrong) | none 0.67 |

Long CV (~10.9 k chars, ~2.4 k tokens, Docker mentioned only in the last line):

| | P(yes) | p50 |
|---|---|---|
| laya multilingual, `max_len=1024` | 0.19 (**truncated**, 1385 tokens dropped) | 417 ms |
| laya multilingual, `max_len=8192` | 0.79 | 1.73 s |
| jevos-v5 (state cache off) | 0.74 | 0.92 s |

Observations (not a benchmark): jevos is very confident (0.96-0.99) on easy items, Laya
multilingual less so; Laya scores look poor zero-shot, as its README warns for base checkpoints.
For full CVs Laya must run with `max_len` >= 4096, otherwise the end of the CV is silently cut.

## Known limits / gotchas

- **Expired HF token**: a stale token in `~/.cache/huggingface/token` makes the (public) Laya
  downloads fail with 401. Use `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` (or refresh the token); laya
  passes `token=None` through so `token=False` cannot be used to disable it.
- Laya: one checkpoint per instance (~2.3-2.8 GB RSS); not thread-parallel here (lock); English
  and typed-decisions checkpoints are ~2.7x slower than multilingual on this CPU.
- jevos: English-only per its authors; 8192-token hard limit (no truncation); instructions and
  option texts <= 8000 chars each; choice <= 26 options, score 2-10 levels; a `choice` with k
  options costs k x (phrasings) prompts, so latency grows with the number of options.
- jevos' state cache (16 states / 8192 tokens) makes repeated questions on the same CV much
  cheaper; disable it (`state_cache=False`) for cold-read latency numbers.
- jevos binary is spawned from `~/.local/share/jevos` (outside the repo) only when nothing
  answers at `base_url`. The stack also runs it as the `jevos` compose service
  (`infra/jevos/Dockerfile`: same release, SHA-256 checked at build time), on 127.0.0.1:8017.
