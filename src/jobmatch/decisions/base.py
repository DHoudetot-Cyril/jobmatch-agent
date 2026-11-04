"""Decision layer contract: "System 1" typed decisions with probabilities.

A decision model answers a closed question about a text `state` and returns a probability
for every option, in one shot, without free-text generation. Backends (Laya, jevos, Qwen
logits via llama.cpp, ...) implement `DecisionModel` and are benchmarked in `bench/decisions/`.

CONTRACT (do not break, the benchmark and the graph depend on it):

    q = Question(kind="yes_no", question="Does the CV show hands-on Docker experience?")
    d = await model.decide(state=cv_text, question=q)
    d.probs   -> {"no": 0.12, "yes": 0.88}   (sums to 1, keys == q.option_labels)
    d.answer  -> "yes"

Kinds:
  - "yes_no":  options are fixed to ["no", "yes"]
  - "choice":  2-16 options given in `options` (label -> description)
  - "score":   ordinal 0..5 levels given in `options` (label -> description), in order
"""

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field, model_validator

DecisionKind = Literal["yes_no", "choice", "score"]


class Question(BaseModel):
    kind: DecisionKind
    question: str = Field(description="The instruction / question to answer about the state")
    options: dict[str, str] = Field(
        default_factory=dict,
        description="label -> description. Ignored for yes_no. Ordered for score.",
    )

    @model_validator(mode="after")
    def _check(self) -> "Question":
        if self.kind == "yes_no":
            self.options = {"no": "No / false", "yes": "Yes / true"}
        elif not 2 <= len(self.options) <= 16:
            raise ValueError("choice/score questions need 2-16 options")
        return self

    @property
    def option_labels(self) -> list[str]:
        return list(self.options)


class Decision(BaseModel):
    probs: dict[str, float]
    latency_ms: float = 0.0
    backend: str = ""

    @property
    def answer(self) -> str:
        return max(self.probs, key=self.probs.__getitem__)

    @property
    def confidence(self) -> float:
        return max(self.probs.values())


class DecisionModel(ABC):
    """A backend. Implementations must be safe to call concurrently from asyncio."""

    name: str

    @abstractmethod
    async def decide(self, state: str, question: Question) -> Decision: ...

    async def aclose(self) -> None:  # optional resource cleanup
        return None
