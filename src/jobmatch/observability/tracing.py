"""Langfuse tracing. CONTRACT (implementation owned by the API task):

    get_langfuse_handler(trace_seed: str | None = None) -> BaseCallbackHandler | None
        LangChain/LangGraph callback handler sending traces to the self-hosted Langfuse,
        or None when `langfuse_enabled` is False or Langfuse is unreachable (never crash
        the app because tracing is down).
        `trace_seed` (optional, e.g. the analysis id) makes the Langfuse trace id
        deterministic (`Langfuse.create_trace_id(seed=...)`), so several runs of the same
        LangGraph thread (first run + resume after the human decision) land in ONE trace.

    flush_langfuse() -> None
        Flush pending events (called on API shutdown and at the end of eval runs).

    langfuse_run_config(analysis_id) -> dict
        `metadata` / `tags` to merge into the LangGraph run config so the trace gets
        session_id = analysis_id, tags ["jobmatch"] and a stable name.

Keys are pre-provisioned in docker-compose via LANGFUSE_INIT_* (see `.env.example`).
SDK >= 3 (here 5.x) is OpenTelemetry based: `from langfuse.langchain import CallbackHandler`;
trace attributes (session, tags, name) are read by the handler from the run config metadata
keys `langfuse_session_id`, `langfuse_tags`, `langfuse_trace_name`.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import httpx
from langchain_core.callbacks import BaseCallbackHandler

from jobmatch.config import get_settings

logger = logging.getLogger(__name__)

TRACE_NAME = "jobmatch-analysis"
TRACE_TAGS = ["jobmatch"]
_HEALTH_TIMEOUT_S = 2.0
_HEALTH_TTL_S = 30.0

_lock = threading.Lock()
_client: Any = None  # langfuse.Langfuse, created lazily
_health: tuple[float, bool] | None = None  # (checked_at, reachable)


def _reachable(host: str) -> bool:
    """Cached, short-timeout check of Langfuse's public health endpoint."""
    global _health
    now = time.monotonic()
    if _health is not None and now - _health[0] < _HEALTH_TTL_S:
        return _health[1]
    try:
        resp = httpx.get(f"{host.rstrip('/')}/api/public/health", timeout=_HEALTH_TIMEOUT_S)
        ok = resp.status_code < 500
    except httpx.HTTPError as exc:
        logger.warning("event=langfuse_unreachable host=%s error=%s", host, exc)
        ok = False
    _health = (now, ok)
    return ok


def _get_client() -> Any:
    global _client
    with _lock:
        if _client is None:
            from langfuse import Langfuse

            s = get_settings()
            _client = Langfuse(
                public_key=s.langfuse_public_key,
                secret_key=s.langfuse_secret_key,
                base_url=s.langfuse_host,
            )
        return _client


def get_langfuse_handler(trace_seed: str | None = None) -> BaseCallbackHandler | None:
    s = get_settings()
    if not s.langfuse_enabled:
        return None
    try:
        if not _reachable(s.langfuse_host):
            return None
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        _get_client()
        trace_context = (
            {"trace_id": Langfuse.create_trace_id(seed=trace_seed)} if trace_seed else None
        )
        return CallbackHandler(public_key=s.langfuse_public_key, trace_context=trace_context)
    except Exception:  # tracing must never break the app
        logger.exception("event=langfuse_handler_error")
        return None


def langfuse_run_config(analysis_id: str) -> dict[str, Any]:
    """Run-config fragment carrying the Langfuse trace attributes for one analysis."""
    return {
        "run_name": TRACE_NAME,
        "tags": list(TRACE_TAGS),
        "metadata": {
            "langfuse_session_id": analysis_id,
            "langfuse_trace_name": TRACE_NAME,
            "langfuse_tags": list(TRACE_TAGS),
            "analysis_id": analysis_id,
        },
    }


def flush_langfuse() -> None:
    client = _client
    if client is None:
        return
    try:
        client.flush()
    except Exception:
        logger.exception("event=langfuse_flush_error")
