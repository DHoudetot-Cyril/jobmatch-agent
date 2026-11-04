"""Shared helper for the structured-output LLM calls made by every agent node.

All agents follow the same pattern (CLAUDE.md contract):

    get_llm().with_structured_output(Model, method="json_schema", include_raw=True)

then count tokens with `record_llm_usage` and, if the reply does not validate against the
schema, retry once with the validation error appended before failing with a clear error.

A reply cut by the token budget (`finish_reason == "length"`, raised by the OpenAI SDK as
`openai.LengthFinishReasonError`) is NOT retried: the same prompt would hit the same budget
again. It fails fast with a `StructuredOutputError` naming the node and the token counts.
"""

import logging
from collections.abc import Sequence

import openai
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel

from jobmatch import llm
from jobmatch.observability.metrics import record_llm_usage

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 2  # first call + one retry


class StructuredOutputError(RuntimeError):
    """The LLM did not return a valid instance of the expected schema."""


def _length_error(node: str, schema: type[BaseModel], exc: BaseException) -> StructuredOutputError:
    usage = getattr(getattr(exc, "completion", None), "usage", None)
    tokens = (
        f"completion_tokens={usage.completion_tokens} prompt_tokens={usage.prompt_tokens}"
        if usage is not None
        else "usage unknown"
    )
    log.error(
        "%s: LLM reply truncated by the token limit (finish_reason=length, %s); "
        "not retrying the identical call",
        node,
        tokens,
    )
    return StructuredOutputError(
        f"{node}: LLM reply for {schema.__name__} was truncated by the token limit "
        f"(finish_reason=length, {tokens})"
    )


async def invoke_structured[M: BaseModel](
    node: str,
    schema: type[M],
    messages: Sequence[BaseMessage],
    *,
    temperature: float | None = None,
    max_tokens: int | None | llm._Default = llm.DEFAULT,
) -> M:
    """Call the LLM with `schema` as structured output; one retry on parsing failure.

    `max_tokens`: see `llm.get_llm` (DEFAULT = settings, None = no cap).
    """
    model = llm.get_llm(temperature=temperature, max_tokens=max_tokens).with_structured_output(
        schema, method="json_schema", include_raw=True
    )
    history = list(messages)
    error: object = None
    for _ in range(MAX_ATTEMPTS):
        try:
            result = await model.ainvoke(history)
        except openai.LengthFinishReasonError as exc:
            raise _length_error(node, schema, exc) from exc
        record_llm_usage(node, result.get("raw"))
        parsed = result.get("parsed")
        if isinstance(parsed, schema):
            return parsed
        error = result.get("parsing_error") or "empty or invalid response"
        if isinstance(error, openai.LengthFinishReasonError):
            raise _length_error(node, schema, error) from error
        history = [
            *messages,
            HumanMessage(
                f"Your previous answer could not be parsed as a valid {schema.__name__} "
                f"({error}). Answer again with JSON that strictly follows the schema."
            ),
        ]
    raise StructuredOutputError(
        f"{node}: LLM output is not a valid {schema.__name__} after {MAX_ATTEMPTS} attempts: "
        f"{error}"
    )
