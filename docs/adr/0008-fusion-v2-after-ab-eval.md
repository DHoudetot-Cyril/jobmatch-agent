# ADR 0008: Skill fusion v2 after the A/B evaluation

- **Status:** accepted (amends the fusion rule of [ADR 0006](0006-decision-layer-jevos.md))
- **Related code:** `src/jobmatch/decisions/service.py` (`fuse_skill`),
  `src/jobmatch/graph/teams/matching.py` (`verify_skills`), `src/jobmatch/schemas.py`
  (`SkillCheck.resolution`), `src/jobmatch/config.py` (`decision_conflict_low`,
  `decision_conflict_high`), `src/jobmatch/graph/teams/writing.py` (`REVIEWER_MAX_TOKENS`).
  Evidence: `evals/results/compare_baseline_vs_jevos.md`,
  `evals/results/compare_stability_off_vs_stability_on.md` (run 2026-11-04, git-ignored; key
  numbers in [`../decision-layer.md`](../decision-layer.md#end-to-end-impact-ab-evaluation)).

## Context

ADR 0006 fused the LLM credit `c` (full 1, partial 0.5, none 0) with jevos' calibrated
P(evidence) `p` as `final = 0.5 * c + 0.5 * p` for every skill. The A/B eval of the full graph
(8 cases, decisions off vs on) showed:

- **Downward bias.** jevos agreed with the LLM on 96 % of the 57 skill checks, yet 6 of the 7
  scored cases went down or stayed equal (up to -18 points on `few_skills_logistics`,
  -9 on `overqualified_support`). A `full` skill keeps credit 1.0 only if `p = 1.0`, which a
  calibrated probability rarely reaches. Must-match skills recall fell from 1.00 to 0.93.
- **Some stability gain.** On the unstable case `few_skills_logistics` the score std fell from
  4.62 (n = 3) to 0.71 (n = 2); the stable case `english_offer_ml` stayed at 56.3 ± 0.58.
- **No quality signal.** Faithfulness 0.925 → 0.943 and relevance 0.713 → 0.743 are within the
  judge noise (about ±0.1).
- **Reviewer.** Approval 12 % → 0 %. jevos checked 11 flagged claims and dismissed none: the
  reviewer was right on facts, its strictness is about length and style. One case failed
  because the reviewer used the whole 4096-token budget (`LengthFinishReasonError`).

## Decision

Replace the average with a rule that keeps the LLM rating unless there is a reason not to:

| LLM rating | Condition | `final_credit` | `resolution` |
|---|---|---|---|
| `partial` | always | `p` | `resolved_partial` |
| `full` | `p < DECISION_CONFLICT_LOW` (0.2) | 0.5 | `conflict` |
| `none` | `p > DECISION_CONFLICT_HIGH` (0.8) | 0.5 | `conflict` |
| `full` / `none` | otherwise | `c` | `llm` |

- Both thresholds are settings. The resolution is stored in every `SkillCheck` and counted in
  `jobmatch_skill_resolution_total`, with a Grafana panel.
- In the same change, the reviewer runs **without a `max_tokens` cap** and a truncated reply is
  no longer retried; offer skills are de-duplicated; the jevos temperatures are applied to
  jevos only.

## Consequences

- **Good:** no systematic bias: when the LLM and jevos agree, the score is the LLM-only score.
- **Good:** the decision model still decides the uncertain cases (`partial`) and tempers strong
  contradictions, and every override is visible (`resolution`, metric).
- **Bad:** where the LLM is firm and wrong but jevos only mildly disagrees (p between 0.2 and
  0.5 for `full`), the LLM error is kept. The stability gain of v1 may shrink, since v1 also
  smoothed confident ratings.
- **Open:** **v2 is not evaluated.** The full suite was not re-run after the change. One smoke
  analysis of the sample documents scored 90 (v1 gave 78-79 on the same sample in earlier
  runs), with all 13 skills resolved `llm` and no reviewer length error. To measure it:
  `make eval ARGS="--label jevos_v2 --decisions on"`, then
  `make eval-compare A=evals/baseline B=evals/results/jevos_v2`.
- **Open:** the reviewer still blocks on length and style issues; `verify_claims` cannot help
  there. A next step is an issue severity (blocking vs minor).
