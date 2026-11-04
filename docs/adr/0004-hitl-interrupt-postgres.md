# ADR 0004: Human-in-the-loop with LangGraph `interrupt` and a Postgres checkpointer

- **Status:** accepted
- **Related code:** `src/jobmatch/graph/human_review.py`, `src/jobmatch/api/service.py`,
  `src/jobmatch/api/main.py`, `src/jobmatch/graph/build.py` (`checkpoint_serializer`)

## Context

A cover letter is sent in the candidate's name, so a human must approve it, ask for changes
or reject it. A full run takes minutes, and the human may answer much later, possibly after
the API was restarted. Re-running the analysis to apply feedback would waste GPU time.

## Decision

- The `human_review` node calls `langgraph.types.interrupt(payload)` with the cover letter and
  the match result. The graph stops there and its state is saved by the checkpointer.
- The checkpointer is `AsyncPostgresSaver` on a dedicated application Postgres (`app-db`),
  separate from Langfuse's database. `thread_id` = analysis id.
- The API resumes with `graph.ainvoke(Command(resume=HumanDecision(...).model_dump()))` on the
  same thread:
  - `approved=true`: status `completed`;
  - `approved=false` + `feedback`: `review` is cleared, `review_iterations` reset, and the
    supervisor sends the run back to the writing team, then to `human_review` again;
  - `approved=false` without feedback: status `rejected`.
- The supervisor sets `status=awaiting_approval` before entering `human_review`, so the API
  can read it while the graph is paused.

## Consequences

- **Good:** a paused analysis survives an API restart and can be approved later; a rewrite
  only re-runs the writing team.
- **Good:** the checkpoint is the single source of truth for the API (`aget_state`); no
  separate status table.
- **Bad:** a run that is **in progress** during a restart is not resumed. The API reports it
  as `failed` ("run stopped unexpectedly") and the user must start a new analysis.
- **Bad:** on resume LangGraph re-executes `human_review` from the top, so the code before
  `interrupt()` must have no side effects.
- **Bad:** Pydantic/Enum values in checkpoints must be allow-listed for deserialization,
  hence `checkpoint_serializer()`; a new type in `jobmatch.schemas` is covered automatically,
  a type defined elsewhere is not.
- Concurrency: a global `asyncio.Lock` serializes decision requests so the same thread cannot
  be resumed twice; a second decision gets HTTP 409.
