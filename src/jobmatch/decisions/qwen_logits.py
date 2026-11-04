"""Qwen via llama-server, decisions read from option-letter probabilities (SemIf-style).

Instead of generating a JSON answer, we label each option with a single-token uppercase letter
(A, B, C, ...), ask for the letter only, and read the next-token distribution in ONE forward
pass (`max_tokens=1`). The probabilities of the option letters are renormalised over the
declared options — the same "read the declared option logits, sample nothing" idea as SemIf
(formerly OpenJev). SemIf itself only scores in-process GGUF/Torch checkpoints, so it cannot
talk to our already-running llama-server; the technique is re-implemented here over HTTP.

Request: POST {base_url}/chat/completions with
    max_tokens=1, temperature=0, logprobs=true, top_logprobs=K,
    chat_template_kwargs={"enable_thinking": false}
`enable_thinking=false` makes the Qwen chat template pre-fill an empty <think></think> block,
so the first generated token is the answer letter (verified on the live server). llama.cpp
reports top logprobs computed BEFORE the sampler chain (post_sampling_probs=false), so top_k /
min_p do not truncate them.

Letters missing from the top-K list get a floor log-probability equal to the smallest logprob
returned (an upper bound on their true value). Token variants ("A", " A") are merged.

Option-order bias: with `permute=True`, each question is also asked with the options in
reversed order and the per-label probabilities are averaged (2 calls per decision; the state
prefix is shared so llama.cpp's prompt cache makes the second call cheap).
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import Any

import httpx

from jobmatch.decisions.base import Decision, DecisionModel, Question

LETTERS = "ABCDEFGHIJKLMNOP"  # 16 = max options allowed by the contract

SYSTEM_PROMPT = (
    "You are a careful, literal classifier. You read a document and answer a closed question "
    "about it by choosing exactly one of the lettered options. Base your answer only on what "
    "the document states. Reply with the option letter only."
)


class DecisionError(RuntimeError):
    """The backend could not produce a decision (HTTP failure or no option letter in logprobs)."""


def build_messages(state: str, question: Question, order: list[str]) -> list[dict[str, str]]:
    """Chat messages for one decision; `order` is the display order of option labels.

    The document comes first so successive questions on the same state share a cached prefix.
    """
    lines = []
    for letter, label in zip(LETTERS, order, strict=False):
        desc = question.options[label]
        lines.append(f"{letter}. {label} — {desc}" if desc else f"{letter}. {label}")
    letters = ", ".join(LETTERS[: len(order)])
    user = (
        f"<document>\n{state.strip()}\n</document>\n\n"
        f"Question: {question.question}\n\n"
        "Options:\n" + "\n".join(lines) + "\n\n"
        f"Answer with a single letter ({letters})."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _normalise_token(tok: str) -> str:
    return tok.strip().strip(".:)(").upper()


def parse_letter_probs(response: dict[str, Any], n_options: int) -> list[float]:
    """Renormalised probabilities of letters A.. (n_options) from a chat completion response."""
    try:
        top = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except (KeyError, IndexError, TypeError) as exc:
        raise DecisionError(f"no logprobs in response: {str(response)[:200]}") from exc
    if not top:
        raise DecisionError("empty top_logprobs")
    wanted = LETTERS[:n_options]
    per_letter: dict[str, list[float]] = {c: [] for c in wanted}
    for entry in top:
        key = _normalise_token(entry.get("token", ""))
        if key in per_letter:
            per_letter[key].append(float(entry["logprob"]))
    if not any(per_letter.values()):
        seen = [e.get("token") for e in top[:5]]
        raise DecisionError(f"no option letter among top logprobs (top tokens: {seen})")
    floor = min(float(e["logprob"]) for e in top)
    logps = [_logsumexp(v) if v else floor for v in (per_letter[c] for c in wanted)]
    m = max(logps)
    exps = [math.exp(lp - m) for lp in logps]
    z = sum(exps)
    return [e / z for e in exps]


def _logsumexp(xs: list[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


class QwenLogitsDecisionModel(DecisionModel):
    name = "qwen-logits"

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        *,
        permute: bool = False,
        top_logprobs: int = 20,
        timeout_s: float = 600.0,
        retries: int = 3,
        max_connections: int = 4,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if base_url is None or model is None:
            from jobmatch.config import get_settings

            s = get_settings()
            base_url = base_url or s.llm_base_url
            model = model or s.llm_model
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.permute = permute
        self.top_logprobs = top_logprobs
        self.retries = retries
        if permute:
            self.name = "qwen-logits-perm"
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s, connect=10.0),
            limits=httpx.Limits(max_connections=max_connections),
            transport=transport,
        )

    def payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
            "chat_template_kwargs": {"enable_thinking": False},
            "cache_prompt": True,
        }

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                r = await self._client.post(f"{self.base_url}/chat/completions", json=body)
                if r.status_code >= 500 or r.status_code == 429:
                    raise DecisionError(f"HTTP {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                return r.json()
            except (httpx.TransportError, DecisionError) as exc:
                last = exc
                if attempt < self.retries:
                    await asyncio.sleep(min(2.0**attempt, 10.0))
        raise DecisionError(f"llama-server request failed after retries: {last}") from last

    async def _probs_for_order(self, state: str, question: Question, order: list[str]):
        resp = await self._post(self.payload(build_messages(state, question, order)))
        letter_probs = parse_letter_probs(resp, len(order))
        return dict(zip(order, letter_probs, strict=True))

    async def decide(self, state: str, question: Question) -> Decision:
        t0 = time.perf_counter()
        labels = question.option_labels
        orders = [labels, labels[::-1]] if self.permute else [labels]
        # sequential on purpose: the second call reuses the first one's cached prompt prefix
        runs = [await self._probs_for_order(state, question, o) for o in orders]
        probs = {lab: sum(r[lab] for r in runs) / len(runs) for lab in labels}
        z = sum(probs.values())
        probs = {k: v / z for k, v in probs.items()}
        return Decision(
            probs=probs, latency_ms=(time.perf_counter() - t0) * 1000, backend=self.name
        )

    async def aclose(self) -> None:
        await self._client.aclose()
