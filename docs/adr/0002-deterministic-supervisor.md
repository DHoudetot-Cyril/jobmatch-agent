# ADR 0002: Deterministic supervisor routing instead of LLM routing

- **Status:** accepted
- **Related code:** `src/jobmatch/graph/supervisor.py`, `src/jobmatch/graph/build.py`

## Context

The graph is hierarchical: a supervisor hands work to three team subgraphs (analysis,
matching, writing) and to a human review step. A common pattern is an LLM supervisor that
reads the state and picks the next agent. Here the pipeline order is fixed:
analysis, then matching, then writing, then human review, with a loop back to writing when
the human asks for changes.

Each LLM call on this hardware is slow (single consumer GPU, 2 slots), and every LLM output
is a potential parsing failure.

## Decision

The supervisor is a **pure function** of the state, `next_step(state)`, with no LLM call:

| Condition | Next |
|---|---|
| status `completed` / `rejected` | `END` |
| `offer` or `profile` missing | `analysis_team` |
| `match` missing | `matching_team` |
| `cover_letter` or `review` missing | `writing_team` |
| otherwise | `human_review` |

Agentic behaviour stays **inside** the teams: parallel parsing, the writer/reviewer loop and
its conditional edge (`route_after_review`).

## Consequences

- **Good:** no extra latency or tokens for routing, and no "wrong or unparsable routing
  decision" failure mode.
- **Good:** routing is unit-testable without an LLM, and the supervisor's `status` updates
  (`running`, `awaiting_approval`) are predictable, which the API relies on.
- **Good:** human feedback is handled by state, not by a prompt: `human_review` clears
  `review`, so the same routing rule sends the run back to the writing team.
- **Bad:** adding a step or changing the order means changing code, not a prompt. The system
  cannot decide on its own to skip a step (for example, not writing a letter for a clear
  mismatch).
- The supervisor still appears in traces and metrics as node `supervisor`.
