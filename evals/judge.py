"""DeepEval judge backed by our local Qwen (llama.cpp `llama-server`).

DeepEval metrics call `generate(prompt, schema=PydanticModel)`; we answer with
`with_structured_output(schema, method="json_schema")` so llama-server constrains the
output to the schema and the judge always returns valid JSON. Without a schema we return
plain text. No cloud LLM is ever involved.
"""

from __future__ import annotations

import asyncio
from typing import Any

from deepeval.models import DeepEvalBaseLLM
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from jobmatch.config import get_settings
from jobmatch.llm import get_llm

JUDGE_TEMPERATURE = 0.0
MAX_ATTEMPTS = 2


class QwenJudge(DeepEvalBaseLLM):
    """`DeepEvalBaseLLM` wrapper around `jobmatch.llm.get_llm()` (temperature 0)."""

    def __init__(self) -> None:
        super().__init__(model=f"local-{get_settings().llm_model}")

    def load_model(self) -> BaseChatModel:
        return get_llm(temperature=JUDGE_TEMPERATURE)

    def get_model_name(self) -> str:
        return f"{get_settings().llm_model} (llama.cpp, local)"

    # Capabilities (prevent DeepEval from asking for logprobs or native JSON mode).
    def supports_log_probs(self) -> bool:
        return False

    def supports_structured_outputs(self) -> bool:
        return True

    def supports_json_mode(self) -> bool:
        return True

    def supports_multimodal(self) -> bool:
        return False

    def generate(self, prompt: str, schema: type[BaseModel] | None = None, **_: Any) -> Any:
        return asyncio.run(self.a_generate(prompt, schema=schema))

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None, **_: Any) -> Any:
        if schema is None:
            msg = await self.model.ainvoke(prompt)
            return str(msg.content)
        structured = self.model.with_structured_output(
            schema, method="json_schema", include_raw=True
        )
        error: object = None
        for _attempt in range(MAX_ATTEMPTS):
            result = await structured.ainvoke(prompt)
            parsed = result.get("parsed")
            if isinstance(parsed, schema):
                return parsed
            error = result.get("parsing_error")
        raise ValueError(f"judge returned no valid {schema.__name__}: {error}")
