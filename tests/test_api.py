"""API tests with a tiny fake graph (real StateGraph + interrupt + InMemorySaver). No LLM/DB."""

import asyncio

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from jobmatch import documents
from jobmatch.api import routes
from jobmatch.api.main import create_app
from jobmatch.api.models import AnalysisView, DependencyCheck
from jobmatch.config import get_settings
from jobmatch.observability import tracing
from jobmatch.schemas import ClaimCheck, CoverLetter, MatchResult, Review
from jobmatch.state import JobMatchState


def _fake_graph(fail: bool = False):
    def score_match(state: JobMatchState) -> dict:
        if fail:
            raise RuntimeError("boom")
        return {
            "match": MatchResult(score=72, summary="Good fit."),
            "trace": ["score_match"],
        }

    def write_letter(state: JobMatchState) -> dict:
        n = state.get("review_iterations", 0) + 1
        return {
            "cover_letter": CoverLetter(subject="Application", body=f"Draft {n}"),
            "review_iterations": n,
            "trace": ["write_letter"],
        }

    def human_review(state: JobMatchState) -> dict:
        decision = interrupt(
            {
                "cover_letter": state["cover_letter"].model_dump(),
                "match": state["match"].model_dump(),
            }
        )
        if decision["approved"]:
            return {"status": "completed", "trace": ["human_review"]}
        if decision.get("feedback"):
            return {"human_feedback": decision["feedback"], "trace": ["human_review"]}
        return {"status": "rejected", "trace": ["human_review"]}

    def route(state: JobMatchState) -> str:
        return END if state.get("status") else "write_letter"

    g = StateGraph(JobMatchState)
    g.add_node("score_match", score_match)
    g.add_node("write_letter", write_letter)
    g.add_node("human_review", human_review)
    g.add_edge(START, "score_match")
    g.add_edge("score_match", "write_letter")
    g.add_edge("write_letter", "human_review")
    g.add_conditional_edges("human_review", route, ["write_letter", END])
    return g.compile(checkpointer=InMemorySaver())


async def _fake_load(data: bytes, filename: str) -> str:
    if filename.endswith(".exe"):
        raise documents.UnsupportedDocumentError("unsupported type")
    return data.decode()


@pytest.fixture(autouse=True)
def _patch_docs(monkeypatch):
    monkeypatch.setattr(documents, "load_document", _fake_load)


async def _client(graph):
    app = create_app(graph=graph, handler_factory=lambda _id: None)
    ctx = app.router.lifespan_context(app)
    await ctx.__aenter__()
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    return app, ctx, client


@pytest.fixture
async def client():
    app, ctx, c = await _client(_fake_graph())
    yield c
    await c.aclose()
    await ctx.__aexit__(None, None, None)


async def _wait_status(client, analysis_id, expected, timeout=5.0):
    async with asyncio.timeout(timeout):
        while True:
            r = await client.get(f"/analyses/{analysis_id}")
            assert r.status_code == 200, r.text
            if r.json()["status"] != "running":
                assert r.json()["status"] == expected, r.json()
                return r.json()
            await asyncio.sleep(0.02)


async def _create(client, **extra):
    files = {"cv": ("cv.txt", b"Python dev")}
    data = {}
    if "offer_file" in extra:
        files["offer"] = extra["offer_file"]
    else:
        data["offer_text"] = extra.get("offer_text", "Senior Python engineer")
    return await client.post("/analyses", files=files, data=data)


async def test_full_flow_approve(client):
    r = await _create(client)
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "running"

    view = await _wait_status(client, body["id"], "awaiting_approval")
    assert view["match"]["score"] == 72
    assert view["pending_review"]["cover_letter"]["body"] == "Draft 1"
    assert view["trace"] == ["score_match", "write_letter"]

    r = await client.post(f"/analyses/{body['id']}/decision", json={"approved": True})
    assert r.status_code == 202
    view = await _wait_status(client, body["id"], "completed")
    assert view["cover_letter"]["body"] == "Draft 1"
    assert view["trace"][-1] == "human_review"
    assert view["pending_review"] is None


async def test_feedback_rewrites_then_reject(client):
    r = await _create(client, offer_file=("offer.md", b"# Offer"))
    aid = r.json()["id"]
    await _wait_status(client, aid, "awaiting_approval")
    r = await client.post(
        f"/analyses/{aid}/decision", json={"approved": False, "feedback": "shorter"}
    )
    assert r.status_code == 202
    view = await _wait_status(client, aid, "awaiting_approval")
    assert view["review_iterations"] == 2
    await client.post(f"/analyses/{aid}/decision", json={"approved": False})
    await _wait_status(client, aid, "rejected")


async def test_unknown_id_404(client):
    assert (await client.get("/analyses/nope")).status_code == 404
    r = await client.post("/analyses/nope/decision", json={"approved": True})
    assert r.status_code == 404


async def test_decision_conflict_409(client):
    r = await _create(client)
    aid = r.json()["id"]
    await _wait_status(client, aid, "awaiting_approval")
    await client.post(f"/analyses/{aid}/decision", json={"approved": True})
    await _wait_status(client, aid, "completed")
    r = await client.post(f"/analyses/{aid}/decision", json={"approved": True})
    assert r.status_code == 409


async def test_unsupported_document_415(client):
    r = await client.post("/analyses", files={"cv": ("cv.exe", b"MZ")}, data={"offer_text": "x"})
    assert r.status_code == 415


async def test_missing_inputs_422(client):
    assert (await client.post("/analyses", data={"offer_text": "x"})).status_code == 422
    r = await client.post("/analyses", files={"cv": ("cv.txt", b"cv")})
    assert r.status_code == 422


async def test_too_large_413(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "max_upload_mb", 0)
    r = await _create(client)
    assert r.status_code == 413


async def test_failed_run():
    app, ctx, c = await _client(_fake_graph(fail=True))
    try:
        aid = (await _create(c)).json()["id"]
        view = await _wait_status(c, aid, "failed")
        assert "boom" in view["error"]
    finally:
        await c.aclose()
        await ctx.__aexit__(None, None, None)


async def test_metrics_use_route_template(client):
    await client.get("/analyses/some-id")
    await client.get("/health")
    text = (await client.get("/metrics")).text
    assert 'jobmatch_http_requests_total{method="GET",route="/analyses/{analysis_id}"' in text
    assert "/analyses/some-id" not in text
    assert 'route="/metrics"' not in text


async def test_health_and_openapi(client):
    assert (await client.get("/health")).json() == {"status": "ok"}
    spec = (await client.get("/openapi.json")).json()
    assert "/analyses/{analysis_id}/decision" in spec["paths"]


def test_tracing_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(get_settings(), "langfuse_enabled", False)
    assert tracing.get_langfuse_handler("abc") is None
    tracing.flush_langfuse()  # never raises


def test_tracing_unreachable_returns_none(monkeypatch):
    monkeypatch.setattr(get_settings(), "langfuse_enabled", True)
    monkeypatch.setattr(get_settings(), "langfuse_host", "http://127.0.0.1:9")
    monkeypatch.setattr(tracing, "_health", None)
    assert tracing.get_langfuse_handler("abc") is None


def test_langfuse_run_config_sets_session():
    cfg = tracing.langfuse_run_config("abc")
    assert cfg["metadata"]["langfuse_session_id"] == "abc"
    assert "jobmatch" in cfg["metadata"]["langfuse_tags"]


# ------------------------------------------------------------------ /ready: decisions check


async def _llm_ok() -> DependencyCheck:
    return DependencyCheck(ok=True, detail="fake")


def _jevos_transport(monkeypatch, handler) -> None:
    real = httpx.AsyncClient

    def client(*args, **kwargs):
        return real(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(routes.httpx, "AsyncClient", client)


async def test_ready_reports_decisions_ok(client, monkeypatch):
    monkeypatch.setattr(routes, "_check_llm", _llm_ok)
    monkeypatch.setattr(get_settings(), "decisions_enabled", True)
    monkeypatch.setattr(get_settings(), "decisions_backend", "jevos")
    _jevos_transport(monkeypatch, lambda req: httpx.Response(200, json={"status": "ready"}))
    r = await client.get("/ready")
    body = r.json()
    assert r.status_code == 200 and body["ready"] is True and body["degraded"] is False
    assert body["checks"]["decisions"]["ok"] is True
    assert body["checks"]["decisions"]["critical"] is False


async def test_ready_stays_ready_when_jevos_is_down(client, monkeypatch):
    monkeypatch.setattr(routes, "_check_llm", _llm_ok)
    monkeypatch.setattr(get_settings(), "decisions_enabled", True)
    monkeypatch.setattr(get_settings(), "decisions_backend", "jevos")

    def down(req):
        raise httpx.ConnectError("refused", request=req)

    _jevos_transport(monkeypatch, down)
    r = await client.get("/ready")
    body = r.json()
    assert r.status_code == 200 and body["ready"] is True and body["degraded"] is True
    assert body["checks"]["decisions"]["ok"] is False
    assert "LLM-only" in body["checks"]["decisions"]["detail"]


async def test_ready_decisions_disabled(client, monkeypatch):
    monkeypatch.setattr(routes, "_check_llm", _llm_ok)
    monkeypatch.setattr(get_settings(), "decisions_enabled", False)
    body = (await client.get("/ready")).json()
    assert body["ready"] is True and body["degraded"] is False
    assert body["checks"]["decisions"]["detail"].startswith("disabled")


async def test_ready_llm_down_is_not_ready(client, monkeypatch):
    async def llm_down() -> DependencyCheck:
        return DependencyCheck(ok=False, detail="down")

    monkeypatch.setattr(routes, "_check_llm", llm_down)
    monkeypatch.setattr(get_settings(), "decisions_enabled", False)
    r = await client.get("/ready")
    assert r.status_code == 503 and r.json()["ready"] is False


# ------------------------------------------------------------------ OpenAPI: review fields


async def test_openapi_exposes_review_decision_fields(client):
    spec = (await client.get("/openapi.json")).json()
    props = spec["components"]["schemas"]["ReviewView"]["properties"]
    assert {"unsupported_claims", "claim_checks"} <= set(props)
    # the LLM-facing schema is unchanged
    assert "claim_checks" not in Review.model_json_schema()["properties"]


def test_analysis_view_serializes_review_decision_fields():
    review = Review(
        approved=False,
        issues=["x"],
        unsupported_claims=["c"],
        claim_checks=[ClaimCheck(claim="c", p_supported=0.1, dismissed=False)],
    )
    view = AnalysisView(id="a", status="completed", review=review)
    dumped = view.model_dump()["review"]
    assert dumped["unsupported_claims"] == ["c"]
    assert dumped["claim_checks"][0]["p_supported"] == 0.1
