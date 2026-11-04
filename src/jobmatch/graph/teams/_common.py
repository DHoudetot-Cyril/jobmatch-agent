"""Helpers shared by the team subgraphs."""

from typing import TypedDict, get_type_hints

from pydantic import BaseModel

from jobmatch.state import JobMatchState

# Input schema of every team subgraph: the full state WITHOUT `trace`.
# `trace` has an append reducer; if a subgraph received the parent's trace it would return it
# whole and the parent would append it a second time. Starting each team with an empty trace
# makes the subgraph return only the nodes it visited.
TeamInput = TypedDict(  # noqa: UP013 - functional form needed to derive the keys
    "TeamInput",
    {k: v for k, v in get_type_hints(JobMatchState, include_extras=True).items() if k != "trace"},
    total=False,
)


def to_json(model: BaseModel | None) -> str:
    """Compact JSON rendering of a domain model for prompts."""
    if model is None:
        return "null"
    return model.model_dump_json(indent=1, exclude_none=True)
