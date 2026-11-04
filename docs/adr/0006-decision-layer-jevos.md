# ADR 0006: Decision layer, jevos only, yes/no only, English questions, calibrated

- **Status:** accepted
- **Related code:** `src/jobmatch/decisions/service.py`, `src/jobmatch/decisions/registry.py`,
  `src/jobmatch/graph/teams/matching.py` (`verify_skills`),
  `src/jobmatch/graph/teams/writing.py` (`verify_claims`), `infra/jevos/Dockerfile`.
  Evidence: `bench/decisions/results/report.md`. Details: [`../decision-layer.md`](../decision-layer.md)

## Context

Two weaknesses of the Qwen-only pipeline: the score varies between runs, and the reviewer is
too strict. In the baseline evaluation it approved 12% of letters and averaged 2.00 review
iterations, the cap (`evals/results/compare_baseline_vs_baseline.md`). Several checks are
closed questions ("does the CV show Docker?", "is this claim supported?") that a small,
calibrated "System 1" model could answer in one shot with a probability.

Four backends implement the `DecisionModel` contract: `jevos`, `laya`, `qwen-logits` and
`qwen-logits-perm`. A benchmark of 194 hand-labelled fictional decisions in French and English
measures accuracy, calibration and automatable share.

## Decision

Use **jevos-v5** (CPU, OpenVINO) and trust it only where the benchmark says so:

- **Yes/no only.** Multiple-choice is unreliable: with English questions, seniority accuracy
  0.538 and doc_type 0.500 (`jevos@q-en`, all languages).
- **English questions**, even for French documents. Claim support on French documents: 0.414
  accuracy with French questions (`jevos`), 1.000 with English questions (`jevos@q-en`, n=29).
  Skill evidence: 0.931 FR (n=58) and 1.000 EN (n=13) with English questions.
  The question templates in `service.py` are copied from the benchmark so the numbers apply.
- **Calibrated.** Temperature scaling fitted on the calib split: T = 0.735 (skill_evidence),
  T = 0.886 (claim_supported), from `bench/decisions/results/summary.json`.
- **Two uses.** `verify_skills` fuses P(evidence) with the LLM credit (50/50);
  `verify_claims` dismisses a reviewer-flagged claim when P(supported) >= 0.8.
- **A/B switch.** `DECISIONS_ENABLED=false` restores the historical LLM-only pipeline.
- **Never fail.** Any error, timeout or too-long CV degrades to the LLM-only result.

**Laya** and **qwen-logits** stay in the repo as alternative backends (registry entries, unit
tests), but they are not used by the graph: they have not been benchmarked
(`bench/decisions/README.md`, "Current scope"), the calibration temperatures are jevos'
values, and Laya's dependencies (`decisions` group) are not installed in the API image.

## Consequences

- **Good:** cheap checks: p50 latency 32-56 ms per decision depending on task and language (`jevos@q-en`) and
  about 1.5 GB peak RSS for `jev serve` (report.md, "Resources"), on CPU, leaving the GPU to Qwen.
- **Good:** each decision is visible in the API output (`skill_checks`, `claim_checks`) and in
  metrics (`jobmatch_decisions_total`, `jobmatch_decision_llm_agreement_total`).
- **Bad:** small test sets (n <= 13 per task in English; Wilson 95% lower bound around 0.65 for
  7/7), so the results separate "usable" from "not usable", not fine differences.
- **Bad:** jevos has an 8192-token context with no truncation; CVs above 24,000 characters skip
  the decision layer.
- **Open:** the end-to-end eval of the graph with the decision layer on
  (`make eval ARGS="--label jevos --decisions on"`) has no committed summary yet.
