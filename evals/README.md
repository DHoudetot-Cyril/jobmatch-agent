# Evaluation suite (DeepEval + local Qwen judge)

Offline evaluation of the full JobMatch graph on a small, **fictional** dataset. Everything
runs locally: the agents and the judge are the same Qwen model served by llama.cpp; DeepEval
telemetry, update checks and Confident AI uploads are disabled (`evals/__init__.py`), and no
OpenAI key is ever used.

## Dataset (`evals/dataset/*.yaml`)

One YAML file per case: `offer_text`, `cv_text` and annotations
`expected_score_range` `[min, max]`, `must_match_skills`, `must_flag_gaps`, `notes`.
An annotation may list alternatives separated by `|` (e.g. `"Cypress|tests e2e"`).

| id | situation | expected score |
|---|---|---|
| `strong_fit_backend` | Python/Django backend dev for a matching offer | 75-100 |
| `partial_fit_devops` | Linux sysadmin vs cloud DevOps (no Kubernetes/Terraform) | 30-65 |
| `mismatch_nurse_data` | nurse vs data scientist (different field) | 0-25 |
| `junior_vs_senior_frontend` | 1-year junior vs lead frontend (7+ years) | 30-65 |
| `overqualified_support` | IT manager (15 years) vs level-1 helpdesk | 65-100 |
| `english_offer_ml` | **English** ML engineer offer, French CV → English letter | 50-85 |
| `employment_gap_accountant` | accountant with a 2-year parental-leave gap | 60-95 |
| `few_skills_logistics` | CV without skills section (warehouse team leader) | 40-80 |

## Metrics

Deterministic (no LLM), computed from the `MatchResult`:

| metric | definition |
|---|---|
| `score_in_range` | 1 if `match.score` ∈ `expected_score_range`, else 0 |
| `skills_recall` | share of `must_match_skills` found in `match.matched_skills` |
| `gaps_recall` | share of `must_flag_gaps` found in `match.gaps[].skill` |

Skill matching is case- and accent-insensitive, whole-word containment either way
(`PostgreSQL` ⊂ `PostgreSQL (Redshift)`) or `difflib` ratio ≥ 0.85.

G-Eval (LLM-as-a-judge, fixed evaluation steps, score 0-10 normalised to 0-1):

| metric | input | actual output | checks |
|---|---|---|---|
| `letter_faithfulness` ★ | CV text | letter | every factual claim is supported by the CV (no hallucination) |
| `letter_relevance` | offer text + expected language | letter | addresses the key requirements, right language, professional |
| `gap_usefulness` | offer + CV + gap list | advice per gap | advice is concrete and actionable |

★ the most important metric: the writer/reviewer loop exists to prevent invented claims.

Thresholds (top of `run_eval.py`) — the run exits with code 1 if an average is below:
`letter_faithfulness >= 0.7`, `score_in_range >= 0.6`.

## Running

Needs llama-server (`~/start_qwen_light.sh`); Langfuse is optional.

```bash
make eval                                            # all cases, label "latest"
make eval ARGS="--cases strong_fit_backend,english_offer_ml"
make eval ARGS="--refresh"                           # re-run the graph, ignore the cache
make eval ARGS="--no-langfuse"                       # no tracing, no scores
make eval ARGS="--label jevos --decisions on"        # named result set, decision layer forced on
# same as: uv run --group eval python -m evals.run_eval [...]
```

| option | effect |
|---|---|
| `--label NAME` | everything goes to `evals/results/<NAME>/` (`outputs/`, `report.md`, `summary.json`); default `latest` |
| `--cases a,b` | run only these cases and **merge** them into the label's existing summary (see below) |
| `--decisions on\|off` | sets `DECISIONS_ENABLED` for the graph run (before the settings are read); default = `.env` / settings |
| `--repeat N` | run the graph N times per case (`outputs/<case>__r<i>.json`) to measure score stability |
| `--judge-all` | with `--repeat`: judge every run (metric = mean over runs) instead of r0 only |
| `--refresh` | ignore cached outputs |
| `--no-langfuse` | no tracing, no scores |

1. **Graph** — each case runs the real graph (in-memory checkpointer) up to the
   `human_review` interrupt, 2 runs at a time (llama-server has 2 slots). The output
   (offer, profile, match, letter, review, `trace` of visited nodes, `decisions_enabled`) is
   cached in `evals/results/<label>/outputs/<case>.json` and reused on the next run, so the
   judges can be re-run cheaply after a prompt/metric change. Delete a file or pass
   `--refresh` to regenerate. **The cache is per label**: never reuse a label for a different
   graph config (the runner refuses to merge `--decisions on` into a `--decisions off` label).
2. **Judging** — deterministic metrics + 3 G-Eval calls per case (2 in parallel).
3. **Outputs** — `evals/results/<label>/report.md` (table per case, averages, threshold
   status, score stability, the judge's reason for every metric) and `summary.json`
   (machine-readable; per case: score, metrics, and `run` = `graph_seconds`,
   `review_iterations`, `reviewer_approved`, plus decision-layer stats when present).
4. **Partial runs** — `--cases` never overwrites a full report: the selected cases replace
   the same cases in the label's `summary.json`, other cases are kept, averages and
   thresholds are recomputed over the merged set and `report.md` is regenerated for all of
   them; `summary.runs` keeps the history of runs merged into the label. Use a fresh
   `--label` for an isolated partial report.
5. **Langfuse** — each graph run is traced with a deterministic trace id
   (`trace_seed = eval-<case>-<timestamp>`, stored in the cached output, tag `eval`), and
   every metric is posted as a score `eval_<metric>` on that trace via
   `POST /api/public/scores` (score id = `<trace>-<metric>`, so re-judging overwrites).
   Skipped silently if Langfuse is down.

## A/B: baseline vs decision layer (jevos)

`evals/baseline/` is a committed snapshot of the Qwen-only graph (before the decision
layer): same layout as a label directory. The new graph adds a "System 1" decision model
(jevos) that verifies each skill credit (`match.skill_checks[]`: `llm_credit`,
`p_evidence`, `final_credit`, `agree`, `resolution`) and each reviewer claim (`review.claim_checks[]`,
`review.unsupported_claims[]`), in the `verify_skills` / `verify_claims` nodes.

```bash
make eval ARGS="--label jevos --decisions on"        # 8 cases, ~25-35 min (GPU + judging)
make eval-compare                                    # = python evals/compare.py evals/baseline evals/results/jevos
uv run --group eval python evals/compare.py A_DIR B_DIR [--out file.md]
```

`compare.py` needs no LLM. It writes `evals/results/compare_<A>_vs_<B>.md` (and prints it):

- judged metrics: average A, average B, Δ = B − A, ↑/↓ with better/worse;
- match score per case (A vs B, flagged when out of the expected range);
- graph behaviour: reviewer approval rate, average review iterations, average graph
  seconds per case, graph errors;
- score stability per case (mean ± std [min-max] (n)) when repeated runs exist;
- decision layer: number of jevos decisions, skill checks and LLM/jevos agreement rate,
  credits overridden, reviewer claims checked / flagged unsupported / dismissed.

Averages are computed over the cases present in both directories. Fields missing from a
snapshot (e.g. `graph_seconds` in the baseline summary) are derived from `outputs/*.json`
(`elapsed_s`, `review_iterations`, `review.approved`); otherwise shown as `n/a`.
`--decisions off` must reproduce the baseline behaviour exactly, so differences between
`baseline` and an `off` run measure the run-to-run noise of the LLM.

### Labels used so far (2026-11-04)

| Label | Config | Graph code |
|---|---|---|
| `evals/baseline/` | decisions off, 8 cases (committed snapshot) | before the decision layer |
| `jevos` | `--decisions on`, 8 cases | decision layer **v1** (50/50 fusion, reviewer capped at 4096 tokens) |
| `stability_off` / `stability_on` | `make eval-stability` (2 cases × 3 runs) | v1 for `on` |

Results and diagnosis: [`docs/decision-layer.md`](../docs/decision-layer.md#end-to-end-impact-ab-evaluation)
(`evals/results/` is git-ignored, so the key numbers are copied there). The current graph uses
**fusion v2** and an uncapped reviewer ([ADR 0008](../docs/adr/0008-fusion-v2-after-ab-eval.md));
**v2 has not been evaluated with this suite yet**. Use a new label, since the cache is per label:

```bash
make eval ARGS="--label jevos_v2 --decisions on"
make eval-compare A=evals/baseline B=evals/results/jevos_v2
make eval-compare A=evals/results/jevos B=evals/results/jevos_v2
```

## Score stability

The graph is not deterministic (sampling, 2 slots in parallel): one run per case cannot tell
a real change from noise. Repeat a few borderline cases and compare the spread:

```bash
make eval-stability            # STAB_CASES=english_offer_ml,few_skills_logistics N=3
# = run_eval --label stability_off --decisions off --repeat 3 --cases ... --no-langfuse
#   run_eval --label stability_on  --decisions on  --repeat 3 --cases ... --no-langfuse
#   compare.py evals/results/stability_off evals/results/stability_on
```

Each run is stored as `outputs/<case>__r<i>.json`; the summary/report give per case the
mean, sample std, min and max of `match.score`. Only r0 is judged (G-Eval) unless
`--judge-all`. Budget ~6.5 min per graph run, 2 in parallel: 2 cases × 3 runs ≈ 20 min per
config (measured 1179 s off, 1195 s on, plus ~2 min of judging), ~40 min for both. A score delta between A and B smaller than the per-case std is
not significant.

## Reading the results

- `score_in_range` failing on several cases = score drift (rubric or scorer prompt issue);
  look at the matched skills / gaps in the report details.
- low `skills_recall` = the scorer did not recognise a skill that is explicit in the CV;
  low `gaps_recall` = a missing skill was rated as matched (over-generous scorer).
- `letter_faithfulness` < 0.7 on a case = the letter very likely contains an invented claim;
  the judge's reason names it. This means the reviewer missed a hallucination.
- The judge is the same model as the agents (self-evaluation bias): treat G-Eval scores as a
  regression signal between versions, not as an absolute quality measure.

## Cost / time

No API cost (local GPU). Measured on an RX 7900 XTX (~37 tok/s, 2 slots), runs of 2026-11-04
(`summary.json`: `graph_seconds`, `judge_seconds`, per-case `elapsed_s`):

| | baseline (off) | jevos v1 (on) |
|---|---|---|
| One graph run, average per case | 386 s | 379 s |
| Graph phase, 8 cases, 2 in parallel | 1141 s (~19 min) | 1569 s (~26 min) |
| Judging, 8 cases | 403 s (~7 min) | 334 s (~6 min) |

So one graph run takes about 6-7 min, and a cold-cache run of the 8 cases about 25-35 min in
total. With a warm cache only the judging runs (~6-7 min for 8 cases). The time cost of the decision
layer is reported by `compare.py` (avg graph seconds per case).
