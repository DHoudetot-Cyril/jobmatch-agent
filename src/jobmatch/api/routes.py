"""HTTP routes."""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from jobmatch import documents
from jobmatch.api.models import (
    AnalysisAccepted,
    AnalysisView,
    DependencyCheck,
    ErrorResponse,
    HealthResponse,
    ReadyResponse,
)
from jobmatch.api.service import (
    AnalysisNotFoundError,
    AnalysisService,
    NotAwaitingApprovalError,
)
from jobmatch.config import get_settings
from jobmatch.decisions.registry import canonical_name
from jobmatch.schemas import AnalysisStatus, HumanDecision

logger = logging.getLogger(__name__)

router = APIRouter()

_READY_TIMEOUT_S = 2.0


def _service(request: Request) -> AnalysisService:
    return request.app.state.service


async def _read_upload(upload: UploadFile, field: str) -> str:
    max_bytes = get_settings().max_upload_mb * 1024 * 1024
    data = await upload.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"'{field}' exceeds {get_settings().max_upload_mb} MB",
        )
    filename = upload.filename or field
    try:
        return await documents.load_document(data, filename)
    except documents.UnsupportedDocumentError as exc:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"'{field}' ({filename}): {exc}"
        ) from exc


@router.post(
    "/analyses",
    tags=["analyses"],
    status_code=status.HTTP_202_ACCEPTED,
    response_model=AnalysisAccepted,
    summary="Start an analysis (CV + job offer)",
    description=(
        "Upload a CV and a job offer (file **or** pasted text). Supported files: .txt, .md, "
        ".pdf (scanned PDFs are OCR'd by the vision model), .png/.jpg/.jpeg/.webp. "
        "The multi-agent graph runs in the background; poll `GET /analyses/{id}`."
    ),
    responses={
        413: {"model": ErrorResponse, "description": "File too large"},
        415: {"model": ErrorResponse, "description": "Unsupported or unreadable document"},
        422: {"description": "Missing CV or offer"},
    },
)
async def create_analysis(
    request: Request,
    cv: Annotated[UploadFile, File(description="Candidate CV")],
    offer: Annotated[UploadFile | None, File(description="Job offer file")] = None,
    offer_text: Annotated[
        str | None, Form(description="Job offer as plain text (if no offer file)")
    ] = None,
) -> AnalysisAccepted:
    if offer is None and not (offer_text and offer_text.strip()):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "Provide either an 'offer' file or 'offer_text'"
        )
    cv_text = await _read_upload(cv, "cv")
    offer_str = await _read_upload(offer, "offer") if offer is not None else offer_text.strip()
    analysis_id = _service(request).start(offer_text=offer_str, cv_text=cv_text)
    return AnalysisAccepted(id=analysis_id)


@router.get(
    "/analyses/{analysis_id}",
    tags=["analyses"],
    response_model=AnalysisView,
    summary="Get an analysis",
    description=(
        "Status is one of `running`, `awaiting_approval` (cover letter waiting for a human "
        "decision, see `pending_review`), `completed`, `rejected`, `failed`."
    ),
    responses={404: {"model": ErrorResponse, "description": "Unknown analysis id"}},
)
async def get_analysis(request: Request, analysis_id: str) -> AnalysisView:
    view = await _service(request).get(analysis_id)
    if view is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Analysis {analysis_id} not found")
    return view


@router.post(
    "/analyses/{analysis_id}/decision",
    tags=["analyses"],
    status_code=status.HTTP_202_ACCEPTED,
    response_model=AnalysisAccepted,
    summary="Approve or reject the cover letter (human-in-the-loop)",
    description=(
        "`approved=true` finishes the analysis. `approved=false` with `feedback` makes the "
        "writer rewrite the letter (then it comes back for approval); without feedback the "
        "analysis is rejected."
    ),
    responses={
        404: {"model": ErrorResponse, "description": "Unknown analysis id"},
        409: {"model": ErrorResponse, "description": "Analysis is not awaiting approval"},
    },
)
async def post_decision(
    request: Request, analysis_id: str, decision: HumanDecision
) -> AnalysisAccepted:
    try:
        await _service(request).decide(analysis_id, decision)
    except AnalysisNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Analysis {analysis_id} not found") from exc
    except NotAwaitingApprovalError as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Analysis {analysis_id} is not awaiting approval (status: {exc.args[0]})",
        ) from exc
    return AnalysisAccepted(id=analysis_id, status=AnalysisStatus.running)


@router.get("/health", tags=["ops"], response_model=HealthResponse, summary="Liveness probe")
async def health() -> HealthResponse:
    return HealthResponse()


async def _check_db(request: Request) -> DependencyCheck:
    pool = getattr(request.app.state, "pool", None)
    if pool is None:
        return DependencyCheck(ok=True, detail="no database pool (in-memory checkpointer)")
    try:
        async with asyncio.timeout(_READY_TIMEOUT_S), pool.connection() as conn:
            await conn.execute("SELECT 1")
        return DependencyCheck(ok=True)
    except Exception as exc:
        return DependencyCheck(ok=False, detail=f"{type(exc).__name__}: {exc}"[:300])


async def _check_llm() -> DependencyCheck:
    base = get_settings().llm_base_url.rstrip("/")
    root = base.removesuffix("/v1")
    errors = []
    async with httpx.AsyncClient(timeout=_READY_TIMEOUT_S) as client:
        for url in (f"{base}/models", f"{root}/health"):
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    return DependencyCheck(ok=True, detail=url)
                errors.append(f"{url} -> HTTP {resp.status_code}")
            except httpx.HTTPError as exc:
                errors.append(f"{url} -> {type(exc).__name__}")
    return DependencyCheck(ok=False, detail="; ".join(errors))


async def _check_decisions() -> DependencyCheck:
    """Decision layer: NON-critical (the graph degrades to LLM-only results without it)."""
    settings = get_settings()
    if not settings.decisions_enabled:
        return DependencyCheck(ok=True, critical=False, detail="disabled (DECISIONS_ENABLED=false)")
    try:
        backend = canonical_name(settings.decisions_backend)
    except KeyError as exc:
        return DependencyCheck(ok=False, critical=False, detail=str(exc))
    if backend != "jevos":
        return DependencyCheck(
            ok=True, critical=False, detail=f"backend {backend!r} (in-process, not probed)"
        )
    url = f"{settings.jevos_url.rstrip('/')}/health"
    try:
        async with httpx.AsyncClient(timeout=_READY_TIMEOUT_S) as client:
            resp = await client.get(url)
        if resp.status_code == 200:
            return DependencyCheck(ok=True, critical=False, detail=url)
        detail = f"{url} -> HTTP {resp.status_code}"
    except httpx.HTTPError as exc:
        detail = f"{url} -> {type(exc).__name__}"
    return DependencyCheck(
        ok=False, critical=False, detail=f"{detail}; analyses degrade to LLM-only results"
    )


@router.get(
    "/ready",
    tags=["ops"],
    response_model=ReadyResponse,
    summary="Readiness probe (database + LLM server; decision layer reported, non-critical)",
    description=(
        "`ready` depends only on the critical checks (database, llm). The `decisions` check "
        "(jevos) is informational: when it fails the API stays ready and `degraded` is true."
    ),
    responses={503: {"model": ReadyResponse, "description": "A critical dependency is down"}},
)
async def ready(request: Request) -> JSONResponse:
    db, llm, dec = await asyncio.gather(_check_db(request), _check_llm(), _check_decisions())
    checks = {"database": db, "llm": llm, "decisions": dec}
    body = ReadyResponse(
        ready=all(c.ok for c in checks.values() if c.critical),
        degraded=any(not c.ok for c in checks.values() if not c.critical),
        checks=checks,
    )
    return JSONResponse(
        body.model_dump(), status_code=200 if body.ready else status.HTTP_503_SERVICE_UNAVAILABLE
    )


@router.get("/metrics", tags=["ops"], summary="Prometheus metrics", include_in_schema=True)
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
