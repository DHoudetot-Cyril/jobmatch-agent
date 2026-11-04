"""LLM factory. CONTRACT: agents get their chat model from `get_llm()` only.

The model is Qwen served by llama.cpp `llama-server` (OpenAI-compatible API, see
`~/start_qwen_light.sh`). Verified features of that server:
  - tool calling (`--jinja`)
  - structured output via `response_format: json_schema`
    -> use `get_llm().with_structured_output(Model, method="json_schema")`
  - vision (`--mmproj`) via OpenAI `image_url` content parts with base64 data URLs
  - Qwen "thinking" is returned separately in `reasoning_content` and is NOT part
    of `message.content`, so parsing `content` is safe.
"""

import base64

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from jobmatch.config import get_settings


def get_llm(temperature: float | None = None) -> BaseChatModel:
    s = get_settings()
    return ChatOpenAI(
        base_url=s.llm_base_url,
        model=s.llm_model,
        api_key=s.llm_api_key,
        temperature=s.llm_temperature if temperature is None else temperature,
        max_tokens=s.llm_max_tokens,
        timeout=s.llm_timeout_s,
        max_retries=2,
    )


def image_message(prompt: str, image_bytes: bytes, mime: str = "image/png") -> HumanMessage:
    """Build a multimodal user message (text + one image) for the vision model."""
    b64 = base64.b64encode(image_bytes).decode()
    return HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]
    )
