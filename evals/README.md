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
make eval                                            # all cases
make eval ARGS="--cases strong_fit_backend,english_offer_ml"
make eval ARGS="--refresh"                           # re-run the graph, ignore the cache
make eval ARGS="--no-langfuse"                       # no tracing, no scores
# same as: uv run --group eval python -m evals.run_eval [...]
```

1. **Graph** — each case runs the real graph (in-memory checkpointer) up to the
   `human_review` interrupt, 2 cases at a time (llama-server has 2 slots). The output
   (offer, profile, match, letter, review) is cached in `evals/results/outputs/<case>.json`
   and reused on the next run, so the judges can be re-run cheaply after a prompt/metric
   change. Delete a file or pass `--refresh` to regenerate.
2. **Judging** — deterministic metrics + 3 G-Eval calls per case (2 in parallel).
3. **Outputs** — `evals/results/report.md` (table per case, averages, threshold status and
   the judge's reason for every metric), `evals/results/summary.json` (machine-readable).
4. **Langfuse** — each graph run is traced with a deterministic trace id
   (`trace_seed = eval-<case>-<timestamp>`, stored in the cached output, tag `eval`), and
   every metric is posted as a score `eval_<metric>` on that trace via
   `POST /api/public/scores` (score id = `<trace>-<metric>`, so re-judging overwrites).
   Skipped silently if Langfuse is down.

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

No API cost (local GPU). On an RX 7900 XTX (~37 tok/s, 2 slots): ~5-6 min per graph run,
so ~20-25 min for the 8 cases with a cold cache, and ~30-60 s of judging per case.
With a warm cache only the judging runs (~5 min for 8 cases).
