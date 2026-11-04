"""Shared helper for the structured-output LLM calls made by every agent node.

All agents follow the same pattern (CLAUDE.md contract):

    get_llm().with_structured_output(Model, method="json_schema", include_raw=True)

then count tokens with `record_llm_usage` and, if the reply does not validate against the
schema, retry once with the validation error appended before failing with a clear error.
"""

from collections.abc import Sequence

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel

from jobmatch import llm
from jobmatch.observability.metrics import record_llm_usage

MAX_ATTEMPTS = 2  # first call + one retry


class StructuredOutputError(RuntimeError):
    """The LLM did not return a valid instance of the expected schema."""


async def invoke_structured[M: BaseModel](
    node: str,
    schema: type[M],
    messages: Sequence[BaseMessage],
    *,
    temperature: float | None = None,
) -> M:
    """Call the LLM with `schema` as structured output; one retry on parsing failure."""
    model = llm.get_llm(temperature=temperature).with_structured_output(
        schema, method="json_schema", include_raw=True
    )
    history = list(messages)
    error: object = None
    for _ in range(MAX_ATTEMPTS):
        result = await model.ainvoke(history)
        record_llm_usage(node, result.get("raw"))
        parsed = result.get("parsed")
        if isinstance(parsed, schema):
            return parsed
        error = result.get("parsing_error") or "empty or invalid response"
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
