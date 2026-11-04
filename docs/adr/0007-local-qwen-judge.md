# ADR 0007: Local Qwen as the DeepEval judge

- **Status:** accepted
- **Related code:** `evals/judge.py`, `evals/metrics.py`, `evals/run_eval.py`,
  [`evals/README.md`](../../evals/README.md)

## Context

The evaluation suite needs an LLM-as-a-judge (G-Eval) for qualities that no deterministic
check can measure: letter faithfulness to the CV (no hallucination), letter relevance to the
offer, and usefulness of the gap advice. DeepEval defaults to a cloud model. The inputs are
CVs and cover letters, and the project's rule is that no data leaves the machine.

## Decision

Wrap the same local Qwen in a `DeepEvalBaseLLM` (`QwenJudge`), at temperature 0, using
`with_structured_output(schema, method="json_schema")` so the judge always returns valid
JSON. DeepEval telemetry, update checks and Confident AI uploads are disabled
(`evals/__init__.py`). G-Eval metrics use fixed evaluation steps; judged metrics sit next to
deterministic ones (`score_in_range`, `skills_recall`, `gaps_recall`).

## Consequences

- **Good (privacy):** CVs and letters never leave the machine; no API key needed.
- **Good (cost):** no per-token cost, so the suite can be re-run after each change; cached
  graph outputs let the judges be re-run alone (about 5 minutes for 8 cases, per
  `evals/README.md`).
- **Bad (self-grading bias):** the judge is the same model as the agents and may rate its own
  style and mistakes too kindly. Baseline faithfulness is 0.925 (`evals/baseline/summary.json`);
  read it as a regression signal between versions, not an absolute quality measure.
- **Bad (noise):** re-judging the same output moves G-Eval scores by about ±0.1 (README,
  "Evaluation").
- **Bad (GPU time):** judging competes with the agents for the two llama-server slots.
- **Next step:** a different judge model would make the evaluation more robust (README,
  "Limitations").
