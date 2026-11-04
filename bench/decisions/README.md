# Decision models benchmark

**What.** A benchmark of "System 1" *decision models*: backends that answer a closed question
about a text (yes/no, or one of N options) in a single shot, returning a **probability per
option** instead of free text. The contract is `src/jobmatch/decisions/base.py`
(`Question` → `Decision.probs`).

**Why.** Several graph nodes ask closed questions that a full JSON-generating LLM call with
"thinking" answers slowly: *does this CV show Docker?*, *is this letter claim supported by the
CV?*, *what seniority is this profile?*, *is this upload a CV or an offer?*. If a decision
model is accurate **and calibrated**, we can automate the confident cases and only send the
uncertain ones to the slow path. This benchmark measures whether that is worth plugging in.

## Backends

| name | how |
|---|---|
| `qwen-logits` | Qwen3.8-27B on our llama-server. Options are labelled A, B, C…; one `/v1/chat/completions` call with `max_tokens=1`, `logprobs=true`, `top_logprobs=20`, `chat_template_kwargs={"enable_thinking": false}`; the letter probabilities are renormalised over the options (SemIf-style option-logit readout). See `src/jobmatch/decisions/qwen_logits.py`. |
| `qwen-logits-perm` | Same, but each question is also asked with the options in reversed order and probabilities are averaged per label (mitigates option-order / letter bias; 2 calls). |
| `laya` | Laya encoder decision model on CPU (`laya_backend.py`, needs `uv sync --group decisions`). |
| `jevos` | jevos decision server (`jevos_backend.py`). |
| `<backend>@q-en` | Ablation run: same backend, but every question and option description is the English version (`question_en` in the dataset); the document is unchanged. Separates "question language" from "document language". |
| `majority-baseline` | Predicts the calib-split class frequencies for every item (sanity floor). |

Backends are created through `jobmatch.decisions.registry.get_backend(name)` (lazy imports, so
torch is only loaded when a CPU backend is requested). A backend that cannot be imported is
skipped with a log line.

## Dataset (`dataset/*.jsonl`, all fictional)

Built by `build_dataset.py` (re-run it after editing labels) from `evals/dataset/*.yaml`,
`data/samples/`, the cover letters in `evals/results/outputs/*.json`, and a few CVs / documents
written for the benchmark. One JSON object per line:

```json
{"id": "...", "task": "skill_evidence", "lang": "fr", "state": "<CV text>",
 "question": {"kind": "yes_no", "question": "..."}, "label": "yes",
 "split": "test", "source": "cv:backend", "note": "why this label"}
```

| task | kind | items | content |
|---|---|---|---|
| `skill_evidence` | yes_no | 102 | CV + "has the person actually used X?" — incl. paraphrases (*conteneurisation* ↔ Docker, OpenShift ↔ Kubernetes, BigQuery ↔ Google Cloud), "notions"/course/hobby-only mentions, homonyms (SWIFT transfers vs Swift, the game of go vs Go, the writer Kafka vs Apache Kafka), absent-but-close skills (MongoDB when PostgreSQL is present). |
| `claim_supported` | yes_no | 52 | CV + one cover-letter sentence. Real sentences from generated letters + crafted subtle falsehoods (wrong numbers, inflated seniority/duration, invented management or collaboration). |
| `seniority` | choice | 20 | junior / mid / senior / lead, with the year ranges spelled out in the option descriptions. |
| `doc_type` | choice | 20 | cv / job_offer / other — "other" includes hard cases (cover letter, #OpenToWork post, job-market article, interview e-mail). |

~20 % of items are English (`lang: en`). Each task is split **stratified by label**: ~30 %
`calib`, ~70 % `test`. Every non-obvious label has a `note`; borderline items are flagged
"spot-check".

## Running

Current scope (lead decision): only **jevos** is benchmarked (`results/report.md`); the
qwen-logits backend is implemented and unit-tested but its benchmark has not been run.

```bash
# jevos (CPU, spawns `jev serve` on demand): ~10 s for the whole dataset
uv run python bench/decisions/run_bench.py --backends jevos,jevos@q-en --concurrency 2
# Qwen (llama-server must be up; 2 slots -> concurrency 2)
uv run python bench/decisions/run_bench.py --backends qwen-logits,qwen-logits-perm --concurrency 2
# CPU backends
uv run --group decisions python bench/decisions/run_bench.py --backends laya,jevos --concurrency 1
# smoke test / partial
uv run python bench/decisions/run_bench.py --backends qwen-logits --tasks doc_type --limit 3
# re-render the report from cached raw results only
uv run python bench/decisions/run_bench.py --backends qwen-logits,qwen-logits-perm,laya,jevos --report-only
```

Each backend runs in its own subprocess (so its load time and peak RSS are isolated) unless
`--in-process`. Raw probabilities are appended to `results/raw/<backend>.jsonl`; a re-run only
computes missing or failed items (delete the file to start over). Outputs:
`results/report.md` (tables + auto-generated observations) and `results/summary.json`.

## Reading the metrics

All quality numbers are on the **test** split, per backend × task × language (`all`, `fr`, `en`).

- **acc / macro-F1** — of the argmax answer. Macro-F1 averages per-class F1, so always answering
  the majority class scores badly (compare with `majority-baseline`).
- **Brier** — multi-class: mean of Σₖ (pₖ − 1[k = gold])²; 0 is perfect, 2 is worst, a uniform
  2-way guess scores 0.5. Rewards both accuracy and honest probabilities.
- **ECE** (10 equal-width bins on the top-label confidence) — average |confidence − accuracy|.
  0.05 means "when it says 90 %, it is right ~85-95 % of the time".
- **raw vs cal** — `cal` applies **temperature scaling** pₖ ∝ pₖ^(1/T), with T fitted by NLL on
  the calib split (per backend × task). T > 1 means the backend was over-confident. It never
  changes the argmax, so acc/F1 are identical raw vs cal.
- **auto@5%** — *automatable share at a 5 % error budget*: sort test items by confidence, answer
  the most confident ones automatically, and take the largest coverage whose error rate stays
  ≤ 5 % (ties are kept together). 0.60 means "60 % of these decisions could be automated with ≤ 5 %
  errors, the rest go to the slow path / a human". The threshold is chosen on the test set itself
  (oracle), so treat it as an optimistic upper bound; with small n it is noisy.
- **latency p50/p95** — per decision, wall clock, as seen by the client (includes queueing
  behind other llama-server users, and 2 calls for `-perm`).
- **Resources** — load time (backend construction + first decision), RSS after load and peak RSS
  of the worker process (`VmHWM`), plus peak RSS of child processes. Qwen runs on the GPU in
  llama-server, so its memory is not visible here.

The report ends with a **Verdict** (bar: on each yes/no task and each language, accuracy ≥ 0.85
and auto@5% ≥ 0.5, with a Wilson 95% lower bound on accuracy for context) and, per backend, the
**10 most confident errors** across all splits — the first place to look for wrong labels.

Caveat: test sets are small (≈14–70 items per task, ≤ 13 in English), so differences of a few
points are noise. The benchmark is meant to separate "clearly usable" from "clearly not".
