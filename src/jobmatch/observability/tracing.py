"""Langfuse tracing. CONTRACT (implementation owned by the API task):

    get_langfuse_handler() -> BaseCallbackHandler | None
        LangChain/LangGraph callback handler sending traces to the self-hosted Langfuse,
        or None when `langfuse_enabled` is False or Langfuse is unreachable (never crash
        the app because tracing is down).

    flush_langfuse() -> None
        Flush pending events (called on API shutdown and at the end of eval runs).

Keys are pre-provisioned in docker-compose via LANGFUSE_INIT_* (see `.env.example`),
so no manual click in the UI is needed. Check the installed `langfuse` SDK version's
docs for the exact import (SDK >= 3 uses `from langfuse.langchain import CallbackHandler`
and reads LANGFUSE_* env vars).
"""

from langchain_core.callbacks import BaseCallbackHandler


def get_langfuse_handler() -> BaseCallbackHandler | None:  # pragma: no cover - stub
    raise NotImplementedError


def flush_langfuse() -> None:  # pragma: no cover - stub
    raise NotImplementedError
