"""jevos backend (feder-cr/jev, MIT): decoder decision model run by OpenVINO on CPU.

jevos is not a Python package: it ships as a C++ binary (`jev`) plus OpenVINO INT8 weights,
served over HTTP in TypeSafe Jev's wire format (`POST /v1/systemone`). See
bench/decisions/BACKENDS.md for the manual install (GitHub release `jevos-v5`).

This backend talks to a running `jev serve`; if none answers at `base_url` and a binary is
found (`binary` arg, `$JEVOS_BIN`, or ~/.local/share/jevos/jev/jev), it spawns one and stops
it in `aclose()`. English only, 8192-token context, no truncation (a longer state is an error).

Mapping of our kinds to jev question types:
  - yes_no -> "noul"   (returns P(yes))       -> {"no": 1 - p, "yes": p}
  - choice -> "choice" with criteria = {label: description} (<= 26 options)
  - score  -> "score"  with criteria = [level text, ...] (2-10 levels, distinct texts);
              probabilities keyed "0".."k-1", mapped back to our labels by position.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from jobmatch.decisions.base import Decision, DecisionModel, Question
from jobmatch.decisions.laya_backend import normalize, option_text

log = logging.getLogger(__name__)

DEFAULT_BINARY = Path.home() / ".local/share/jevos/jev/jev"
MAX_SCORE_LEVELS = 10


def to_jev_question(question: Question) -> dict[str, Any]:
    """Our Question -> one jev question definition."""
    if question.kind == "yes_no":
        return {"type": "noul", "instructions": question.question}
    if question.kind == "choice":
        return {
            "type": "choice",
            "instructions": question.question,
            "criteria": {lab: (desc or None) for lab, desc in question.options.items()},
        }
    if len(question.options) > MAX_SCORE_LEVELS:
        raise ValueError(f"jevos scores take 2-{MAX_SCORE_LEVELS} levels")
    return {
        "type": "score",
        "instructions": question.question,
        "criteria": [option_text(lab, d) for lab, d in question.options.items()],
    }


def from_jev_answer(question: Question, answer: dict[str, Any]) -> dict[str, float]:
    """One jev answer -> probs over our option labels."""
    labels = question.option_labels
    if question.kind == "yes_no":
        p = min(1.0, max(0.0, float(answer["noul"])))
        return {"no": 1.0 - p, "yes": p}
    raw = answer["probabilities"]
    if question.kind == "choice":
        return normalize({lab: raw.get(lab, 0.0) for lab in labels}, labels)
    return normalize({lab: raw.get(str(i), 0.0) for i, lab in enumerate(labels)}, labels)


class JevosDecisionModel(DecisionModel):
    name = "jevos"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8017",
        *,
        binary: str | os.PathLike[str] | None = None,
        threads: int | None = None,
        state_cache: bool = True,
        timeout: float = 120.0,
        start_timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        env_bin = os.environ.get("JEVOS_BIN")
        self.binary = Path(binary or env_bin or DEFAULT_BINARY).expanduser()
        self.threads = threads
        self.state_cache = state_cache
        self.start_timeout = start_timeout
        self.load_seconds: float | None = None
        self.model_name: str | None = None
        self.timeout = timeout
        self._client_obj: httpx.AsyncClient | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._ready = False
        self._start_lock = asyncio.Lock()

    @property
    def _client(self) -> httpx.AsyncClient:
        if self._client_obj is None:
            self._client_obj = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self._client_obj

    async def _health(self) -> dict[str, Any] | None:
        try:
            r = await self._client.get("/health", timeout=2.0)
        except httpx.TransportError:
            return None
        if r.status_code != 200:
            return None
        body = r.json()
        return body if body.get("status") == "ready" else None

    def _spawn(self) -> subprocess.Popen[bytes]:
        if not self.binary.is_file():
            raise RuntimeError(
                f"jevos: no server at {self.base_url} and no `jev` binary at {self.binary}. "
                "Install it from https://github.com/feder-cr/jev/releases/tag/jevos-v5 "
                "(see bench/decisions/BACKENDS.md) or set $JEVOS_BIN."
            )
        url = urlparse(self.base_url)
        cmd = [
            str(self.binary),
            "serve",
            "--host",
            url.hostname or "127.0.0.1",
            "--port",
            str(url.port or 8017),
        ]
        if self.threads:
            cmd += ["--threads", str(self.threads)]
        if not self.state_cache:
            cmd += ["--state-cache", "0"]
        log.info("jevos: starting %s", " ".join(cmd))
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    async def _ensure_ready(self) -> None:
        if self._ready:
            return
        async with self._start_lock:
            if self._ready:
                return
            t0 = time.perf_counter()
            health = await self._health()
            if health is None:
                self._proc = self._spawn()
                deadline = t0 + self.start_timeout
                while health is None:
                    if self._proc.poll() is not None:
                        raise RuntimeError(f"jevos: `jev serve` exited ({self._proc.returncode})")
                    if time.perf_counter() > deadline:
                        raise RuntimeError("jevos: server not ready before start_timeout")
                    await asyncio.sleep(0.2)
                    health = await self._health()
            self.load_seconds = time.perf_counter() - t0
            self.model_name = health.get("model")
            self._ready = True

    async def decide(self, state: str, question: Question) -> Decision:
        payload = {
            "model": "jev-latest",
            "state": state,
            "questions": {"q": to_jev_question(question)},
        }
        await self._ensure_ready()
        t0 = time.perf_counter()
        r = await self._client.post("/v1/systemone", json=payload)
        if r.status_code != 200:
            raise RuntimeError(f"jevos: HTTP {r.status_code}: {r.text[:500]}")
        probs = from_jev_answer(question, r.json()["answers"]["q"])
        return Decision(
            probs=probs, latency_ms=(time.perf_counter() - t0) * 1000, backend=self.name
        )

    async def aclose(self) -> None:
        if self._client_obj is not None:
            await self._client_obj.aclose()
            self._client_obj = None
        if self._proc is not None:
            self._proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._proc.wait(timeout=10)
            if self._proc.poll() is None:
                self._proc.kill()
            self._proc = None
        self._ready = False
