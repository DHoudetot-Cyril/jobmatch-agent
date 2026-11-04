"""Analysis lifecycle: start / inspect / resume LangGraph runs in background asyncio tasks.

One analysis = one LangGraph thread (thread_id = analysis id) = one Langfuse trace & session.
The checkpointer is the source of truth; in-memory state only tracks running tasks (and a
fallback for failures that could not be written to the checkpoint).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.types import Command

from jobmatch.api.models import AnalysisView
from jobmatch.observability.metrics import (
    ANALYSES,
    ANALYSES_IN_PROGRESS,
    MATCH_SCORE,
    REVIEW_ITERATIONS,
)
from jobmatch.observability.tracing import get_langfuse_handler, langfuse_run_config
from jobmatch.schemas import AnalysisStatus, HumanDecision

logger = logging.getLogger(__name__)

HandlerFactory = Callable[[str], BaseCallbackHandler | None]

_FINAL = {AnalysisStatus.completed, AnalysisStatus.rejected, AnalysisStatus.failed}


class NotAwaitingApprovalError(Exception):
    """A decision was posted for an analysis that is not paused on human review."""


class AnalysisNotFoundError(Exception):
    pass


def _get(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


class AnalysisService:
    def __init__(self, graph: Any, handler_factory: HandlerFactory | None = None) -> None:
        self.graph = graph
        self._handler_factory = handler_factory or get_langfuse_handler
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._failures: dict[str, str] = {}
        self._decision_lock = asyncio.Lock()

    # ------------------------------------------------------------------ public API
    def start(self, offer_text: str, cv_text: str) -> str:
        analysis_id = str(uuid.uuid4())
        self._spawn(analysis_id, {"offer_text": offer_text, "cv_text": cv_text}, phase="start")
        logger.info(
            "event=analysis_started analysis_id=%s offer_chars=%d cv_chars=%d",
            analysis_id,
            len(offer_text),
            len(cv_text),
        )
        return analysis_id

    async def get(self, analysis_id: str) -> AnalysisView | None:
        snapshot = await self.graph.aget_state(self._thread(analysis_id))
        values: dict[str, Any] = dict(snapshot.values or {})
        running = self.is_running(analysis_id)
        if not values and not running and analysis_id not in self._failures:
            return None

        error = values.get("error") or self._failures.get(analysis_id)
        interrupts = getattr(snapshot, "interrupts", ()) or ()
        if analysis_id in self._failures and not running:
            status = AnalysisStatus.failed
        else:
            status = self._status(values, snapshot, running, interrupts)
        if status is AnalysisStatus.failed and not error:
            error = "run stopped unexpectedly (e.g. API restart); please start a new analysis"

        return AnalysisView(
            id=analysis_id,
            status=status,
            offer=values.get("offer"),
            profile=values.get("profile"),
            match=values.get("match"),
            career_plan=values.get("career_plan"),
            cover_letter=values.get("cover_letter"),
            review=values.get("review"),
            review_iterations=values.get("review_iterations") or 0,
            trace=values.get("trace") or [],
            pending_review=(
                interrupts[0].value
                if status is AnalysisStatus.awaiting_approval and interrupts
                else None
            ),
            error=error,
        )

    async def decide(self, analysis_id: str, decision: HumanDecision) -> None:
        async with self._decision_lock:
            view = await self.get(analysis_id)
            if view is None:
                raise AnalysisNotFoundError(analysis_id)
            if view.status is not AnalysisStatus.awaiting_approval:
                raise NotAwaitingApprovalError(view.status)
            self._spawn(analysis_id, Command(resume=decision.model_dump()), phase="resume")
        logger.info(
            "event=analysis_resumed analysis_id=%s approved=%s has_feedback=%s",
            analysis_id,
            decision.approved,
            bool(decision.feedback),
        )

    def is_running(self, analysis_id: str) -> bool:
        task = self._tasks.get(analysis_id)
        return task is not None and not task.done()

    async def shutdown(self) -> None:
        tasks = [t for t in self._tasks.values() if not t.done()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _thread(analysis_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": analysis_id}}

    @staticmethod
    def _status(
        values: dict[str, Any], snapshot: Any, running: bool, interrupts: Any
    ) -> AnalysisStatus:
        if running:
            return AnalysisStatus.running
        raw = values.get("status")
        status = AnalysisStatus(raw) if raw else None
        if status is AnalysisStatus.failed:
            return status
        if interrupts:
            return AnalysisStatus.awaiting_approval
        if status in _FINAL:
            return status
        if not snapshot.next:
            return AnalysisStatus.completed
        return AnalysisStatus.failed

    def _spawn(self, analysis_id: str, payload: Any, phase: str) -> None:
        task = asyncio.create_task(self._run(analysis_id, payload, phase), name=analysis_id)
        self._tasks[analysis_id] = task  # strong ref: prevents GC of the running task
        self._failures.pop(analysis_id, None)

    def _run_config(self, analysis_id: str, handler: BaseCallbackHandler | None) -> dict:
        return {
            **langfuse_run_config(analysis_id),
            **self._thread(analysis_id),
            "callbacks": [handler] if handler else [],
        }

    async def _run(self, analysis_id: str, payload: Any, phase: str) -> None:
        ANALYSES_IN_PROGRESS.inc()
        started = time.perf_counter()
        try:
            handler = await asyncio.to_thread(self._handler_factory, analysis_id)
            await self.graph.ainvoke(payload, config=self._run_config(analysis_id, handler))
            await self._after_run(analysis_id, phase, time.perf_counter() - started)
        except Exception as exc:
            logger.exception(
                "event=analysis_failed analysis_id=%s phase=%s error=%r", analysis_id, phase, exc
            )
            await self._mark_failed(analysis_id, f"{type(exc).__name__}: {exc}"[:1000])
        finally:
            ANALYSES_IN_PROGRESS.dec()

    async def _after_run(self, analysis_id: str, phase: str, duration: float) -> None:
        snapshot = await self.graph.aget_state(self._thread(analysis_id))
        values = snapshot.values or {}
        if getattr(snapshot, "interrupts", ()):
            logger.info(
                "event=analysis_awaiting_approval analysis_id=%s phase=%s duration_s=%.2f",
                analysis_id,
                phase,
                duration,
            )
            return
        status = AnalysisStatus(values.get("status") or AnalysisStatus.completed)
        ANALYSES.labels(status=status.value).inc()
        score = _get(values.get("match"), "score")
        if score is not None:
            MATCH_SCORE.observe(score)
        REVIEW_ITERATIONS.observe(values.get("review_iterations") or 0)
        logger.info(
            "event=analysis_finished analysis_id=%s status=%s score=%s duration_s=%.2f",
            analysis_id,
            status.value,
            score,
            duration,
        )

    async def _mark_failed(self, analysis_id: str, error: str) -> None:
        ANALYSES.labels(status=AnalysisStatus.failed.value).inc()
        self._failures[analysis_id] = error
        try:
            await self.graph.aupdate_state(
                self._thread(analysis_id), {"status": AnalysisStatus.failed, "error": error}
            )
        except Exception as exc:  # keep the in-memory fallback
            logger.warning(
                "event=failed_status_not_persisted analysis_id=%s error=%r", analysis_id, exc
            )
