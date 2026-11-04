"""Laya backend (convaiinnovations/laya, Apache-2.0): encoder decision model, one forward pass.

Install with `uv sync --group decisions` (CPU-only torch). Checkpoints are downloaded from the
Hugging Face Hub on first use, pinned to the commit SHAs the laya package reviewed.

Mapping of our kinds to Laya question types:
  - yes_no -> "noul"   (Laya returns P(true))  -> {"no": 1 - p, "yes": p}
  - choice -> "choice" with criteria = {label: description}
  - score  -> "score"  with criteria = [level text, ...] in our option order; Laya returns
              probabilities keyed "0".."k-1", mapped back to our labels by position.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

from jobmatch.decisions.base import Decision, DecisionModel, Question

log = logging.getLogger(__name__)

# name -> (hub repo, reviewed revision). Revisions = laya 0.4.22 `PINNED_REVISIONS`.
CHECKPOINTS: dict[str, tuple[str, str | None]] = {
    "multilingual": (
        "convaiinnovations/laya-multilingual",
        "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67",
    ),
    "english": ("convaiinnovations/laya", "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"),
    "typed-decisions": (
        "convaiinnovations/laya-typed-decisions",
        "1a793eb568e6718f15941d08f85432581df534e3",
    ),
}


def option_text(label: str, description: str) -> str:
    """Text of one ordered level: 'label: description' (or just the label)."""
    description = (description or "").strip()
    if not description or description == label:
        return label
    return f"{label}: {description}"


def normalize(probs: dict[str, float], labels: list[str]) -> dict[str, float]:
    """Re-normalise over `labels` (backends round their outputs); uniform if degenerate."""
    values = [max(0.0, float(probs.get(lab, 0.0))) for lab in labels]
    total = sum(values)
    if total <= 0.0:
        return {lab: 1.0 / len(labels) for lab in labels}
    return {lab: v / total for lab, v in zip(labels, values, strict=True)}


def to_laya_question(question: Question) -> dict[str, Any]:
    """Our Question -> one Laya question definition."""
    if question.kind == "yes_no":
        return {"type": "noul", "instructions": question.question}
    if question.kind == "choice":
        return {
            "type": "choice",
            "instructions": question.question,
            "criteria": {lab: (desc or None) for lab, desc in question.options.items()},
        }
    return {
        "type": "score",
        "instructions": question.question,
        "criteria": [option_text(lab, d) for lab, d in question.options.items()],
    }


def from_laya_answer(question: Question, answer: dict[str, Any]) -> dict[str, float]:
    """One Laya answer -> probs over our option labels."""
    labels = question.option_labels
    if question.kind == "yes_no":
        p = min(1.0, max(0.0, float(answer["noul"])))
        return {"no": 1.0 - p, "yes": p}
    raw = answer["probabilities"]
    if question.kind == "choice":
        return normalize({lab: raw.get(lab, 0.0) for lab in labels}, labels)
    return normalize({lab: raw.get(str(i), 0.0) for i, lab in enumerate(labels)}, labels)


class LayaDecisionModel(DecisionModel):
    """Laya on CPU. `checkpoint` is a name from CHECKPOINTS, a Hub repo id or a local directory."""

    name = "laya"

    def __init__(
        self,
        checkpoint: str = "multilingual",
        *,
        max_len: int | None = 1024,
        head_max_len: int | None = None,
        revision: str | None = None,
        device: str = "cpu",
        num_threads: int | None = None,
    ) -> None:
        if max_len is not None and not 64 <= max_len <= 8192:
            raise ValueError("max_len must be in [64, 8192] (None = checkpoint default)")
        repo, pinned = CHECKPOINTS.get(checkpoint, (checkpoint, None))
        self.checkpoint = checkpoint
        self.repo = repo
        self.revision = revision or pinned
        self.max_len = max_len
        self.head_max_len = head_max_len
        self.device = device
        self.num_threads = num_threads
        self.load_seconds: float | None = None
        self._agent: Any = None
        self._load_lock = threading.Lock()
        # torch already uses every core for one forward pass: serialise calls.
        self._infer_lock = threading.Lock()

    def _load(self) -> Any:
        with self._load_lock:
            if self._agent is None:
                try:
                    import laya
                    import torch
                except ImportError as e:  # pragma: no cover - depends on the env
                    raise RuntimeError(
                        "laya is not installed: run `uv sync --group decisions`"
                    ) from e
                if self.num_threads:
                    torch.set_num_threads(self.num_threads)
                t0 = time.perf_counter()
                self._agent = laya.load(self.repo, device=self.device, revision=self.revision)
                self.load_seconds = time.perf_counter() - t0
                log.info("laya %s loaded in %.1fs", self.repo, self.load_seconds)
        return self._agent

    def _predict(self, state: str, question: Question) -> dict[str, float]:
        agent = self._load()
        with self._infer_lock:
            result = agent.system_one(
                state,
                {"q": to_laya_question(question)},
                max_len=self.max_len,
                head_max_len=self.head_max_len,
            )
        usage = result.get("usage") or {}
        if usage.get("truncated"):
            log.warning(
                "laya: state truncated (%s tokens dropped, max_len=%s)",
                usage.get("state_tokens_dropped"),
                self.max_len,
            )
        return from_laya_answer(question, result["answers"]["q"])

    async def decide(self, state: str, question: Question) -> Decision:
        t0 = time.perf_counter()
        probs = await asyncio.to_thread(self._predict, state, question)
        return Decision(
            probs=probs, latency_ms=(time.perf_counter() - t0) * 1000, backend=self.name
        )

    async def aclose(self) -> None:
        self._agent = None
