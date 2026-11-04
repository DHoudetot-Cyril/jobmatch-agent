# ADR 0003: Match score computed in code from per-skill LLM ratings

- **Status:** accepted
- **Related code:** `src/jobmatch/graph/teams/matching.py`, `src/jobmatch/schemas.py`
  (`ScoreAssessment`, `SkillAssessment`), `src/jobmatch/prompts/matching.py`

## Context

Asking an LLM "give a 0-100 match score" produces a number that cannot be explained, is hard
to test, and drifts between prompts and runs. The score is the main output a user looks at,
so it needs to be justified skill by skill.

## Decision

`score_match` asks the LLM for a `ScoreAssessment`, not a score:

- for **every** offer skill: `match` = `full` / `partial` / `none` and the CV `evidence`;
- an `experience_fit` rating from 0 to 100, plus strengths and a summary.

The code then computes the score with a fixed rubric (`weighted_score`):

| Component | Weight | Credit |
|---|---|---|
| Required skills | 0.70 | mean of `full`=1, `partial`=0.5, `none`=0 |
| Nice-to-have skills | 0.15 | same |
| Experience fit | 0.15 | `experience_fit / 100` |

Weights of an empty category are redistributed. Skill levels are taken from the parsed offer,
not from the LLM's copy. The same function is reused by `verify_skills` with fused credits
(`0.5 * llm_credit + 0.5 * p_evidence`, see [ADR 0006](0006-decision-layer-jevos.md)).

## Consequences

- **Good:** every point of the score traces back to a skill rating and a CV quote, which the
  API returns (`matched_skills`, `gaps`, `skill_checks`).
- **Good:** the rubric is unit-tested without an LLM, and a second opinion (the decision
  model) can be fused per skill.
- **Good:** in the baseline evaluation, 8/8 cases fall in their expected score range, from 8
  (mismatch) to 96 (`evals/baseline/summary.json`, `score_in_range` = 1.0).
- **Bad:** the score still depends on the LLM's ratings, so it can vary by a few points
  between runs (README, "Limitations"); `make eval-stability` measures it.
- **Bad:** the weights are hand-chosen, not learned, and all skills of a level weigh the same.
