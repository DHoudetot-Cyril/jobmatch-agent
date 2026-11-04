"""Prometheus metrics. CONTRACT: metric names below are used by the Grafana dashboard.

- Graph nodes are wrapped with `@instrument_node("node_name")` (works on sync and async fns).
- After each LLM call, nodes call `record_llm_usage(node, ai_message_or_raw)` to count tokens.
- The API owns the HTTP metrics and the analysis-level metrics.
- llama-server metrics (`llamacpp:*`) are scraped directly by Prometheus, not defined here.
"""

import functools
import inspect
import time
from collections.abc import Callable
from typing import Any

from prometheus_client import Counter, Gauge, Histogram

# --- HTTP (API) ---
HTTP_REQUESTS = Counter(
    "jobmatch_http_requests_total", "HTTP requests", ["method", "route", "status"]
)
HTTP_LATENCY = Histogram(
    "jobmatch_http_request_duration_seconds", "HTTP request latency", ["method", "route"]
)

# --- Agents ---
NODE_LATENCY = Histogram(
    "jobmatch_agent_node_duration_seconds",
    "Duration of one graph node execution",
    ["node"],
    buckets=(0.1, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300),
)
NODE_ERRORS = Counter("jobmatch_agent_node_errors_total", "Graph node failures", ["node"])
LLM_TOKENS = Counter(
    "jobmatch_llm_tokens_total", "LLM tokens used", ["node", "kind"]  # kind: prompt|completion
)

# --- Analyses (business) ---
ANALYSES = Counter(
    "jobmatch_analyses_total", "Finished analyses", ["status"]  # completed|rejected|failed
)
ANALYSES_IN_PROGRESS = Gauge("jobmatch_analyses_in_progress", "Analyses currently running")
MATCH_SCORE = Histogram(
    "jobmatch_match_score",
    "Match score distribution",
    buckets=(10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
)
REVIEW_ITERATIONS = Histogram(
    "jobmatch_review_iterations", "Writer/reviewer loops per analysis", buckets=(0, 1, 2, 3, 4, 5)
)


def instrument_node(node: str) -> Callable:
    """Decorator measuring latency and errors of a graph node (sync or async)."""

    def deco(fn: Callable) -> Callable:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                try:
                    return await fn(*args, **kwargs)
                except Exception:
                    NODE_ERRORS.labels(node=node).inc()
                    raise
                finally:
                    NODE_LATENCY.labels(node=node).observe(time.perf_counter() - start)

            return awrapper

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            except Exception:
                NODE_ERRORS.labels(node=node).inc()
                raise
            finally:
                NODE_LATENCY.labels(node=node).observe(time.perf_counter() - start)

        return wrapper

    return deco


def record_llm_usage(node: str, message: Any) -> None:
    """Count tokens from an AIMessage (`usage_metadata`) or a structured-output raw dict.

    With `with_structured_output(..., include_raw=True)`, pass `result["raw"]`.
    Silently ignores objects without usage info.
    """
    usage = getattr(message, "usage_metadata", None)
    if not usage:
        return
    LLM_TOKENS.labels(node=node, kind="prompt").inc(usage.get("input_tokens", 0))
    LLM_TOKENS.labels(node=node, kind="completion").inc(usage.get("output_tokens", 0))
