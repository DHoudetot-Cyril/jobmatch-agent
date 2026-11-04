"""FastAPI application: `uvicorn jobmatch.api.main:app`."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from jobmatch.api.routes import router
from jobmatch.api.service import AnalysisService, HandlerFactory
from jobmatch.config import get_settings
from jobmatch.observability.metrics import HTTP_LATENCY, HTTP_REQUESTS
from jobmatch.observability.tracing import flush_langfuse

logger = logging.getLogger("jobmatch.api")

DESCRIPTION = """
Hierarchical LangGraph multi-agent system matching a **CV** to a **job offer**:
structured parsing, match score & skill gaps, cover letter written by a writer/reviewer loop,
then **human-in-the-loop** approval.

Flow: `POST /analyses` → poll `GET /analyses/{id}` until `awaiting_approval` →
`POST /analyses/{id}/decision` → poll until `completed` / `rejected`.

Each analysis is one Langfuse trace (session id = analysis id); metrics on `/metrics`.
"""

TAGS = [
    {"name": "analyses", "description": "Run and drive CV / job-offer analyses."},
    {"name": "ops", "description": "Health, readiness and Prometheus metrics."},
]


def configure_logging() -> None:
    root = logging.getLogger()
    if any(getattr(h, "_jobmatch", False) for h in root.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s level=%(levelname)s logger=%(name)s %(message)s")
    )
    handler._jobmatch = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())


class HttpMetricsMiddleware:
    """Counts requests / latency labelled by the matched ROUTE TEMPLATE (low cardinality)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") == "/metrics":
            await self.app(scope, receive, send)
            return
        start = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            route = scope.get("route")
            template = getattr(route, "path", None) or "unmatched"
            method = scope.get("method", "GET")
            HTTP_REQUESTS.labels(method=method, route=template, status=str(status_code)).inc()
            HTTP_LATENCY.labels(method=method, route=template).observe(time.perf_counter() - start)


async def _open_postgres_graph(app: FastAPI) -> Any:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    from jobmatch.graph.build import build_graph, checkpoint_serializer

    pool = AsyncConnectionPool(
        get_settings().database_url,
        open=False,
        max_size=10,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    await pool.open(wait=True, timeout=15)
    app.state.pool = pool
    saver = AsyncPostgresSaver(pool, serde=checkpoint_serializer())
    await saver.setup()
    logger.info("event=checkpointer_ready backend=postgres")
    return build_graph(checkpointer=saver)


def create_app(graph: Any | None = None, handler_factory: HandlerFactory | None = None) -> FastAPI:
    """Build the app. Pass `graph` (compiled, with its own checkpointer) to skip Postgres
    (tests); `handler_factory(analysis_id)` overrides the Langfuse handler factory."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        app.state.pool = None
        compiled = graph if graph is not None else await _open_postgres_graph(app)
        app.state.service = AnalysisService(compiled, handler_factory=handler_factory)
        logger.info("event=api_started")
        try:
            yield
        finally:
            await app.state.service.shutdown()
            if app.state.pool is not None:
                await app.state.pool.close()
            await asyncio.to_thread(flush_langfuse)
            logger.info("event=api_stopped")

    app = FastAPI(
        title="JobMatch Agent API",
        version="0.1.0",
        description=DESCRIPTION,
        openapi_tags=TAGS,
        lifespan=lifespan,
    )
    app.add_middleware(HttpMetricsMiddleware)
    app.include_router(router)
    return app


app = create_app()
