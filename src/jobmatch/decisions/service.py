"""Decision service used by the graph: calibrated yes/no checks ("System 1", ~50 ms on CPU).

The graph never talks to a backend directly; it calls the two functions below. Both are
GRACEFULLY DEGRADING: if the decision layer is disabled, unreachable or errors, they log
(a warning the first time, then debug), count the failure in Prometheus and return `[]`;
callers then keep the LLM-only result untouched. An analysis never fails because of them.

    check_skill_evidence(cv_text, assessments) -> list[SkillCheck]   (matching team, fusion v2:
        the LLM rating is kept unless "partial" (-> p) or strongly contradicted (-> 0.5),
        see `fuse_skill`)
    check_claims(cv_text, claims)              -> list[ClaimCheck]   (writing team)

What jevos is trusted with comes from the benchmark (bench/decisions/results/report.md):
only yes/no questions, phrased in ENGLISH (documents may stay French). With English questions
(`jevos@q-en`): skill_evidence acc 0.93 FR / 1.00 EN, claim_supported 1.00 FR / 1.00 EN; with
French questions claim_supported drops to 0.41. Multiple-choice is unreliable -> never used.
The question templates below are copied verbatim from bench/decisions/build_dataset.py
(`SKILL_Q["en"]`, `CLAIM_Q["en"]`) so that the benchmark numbers apply.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Sequence

from jobmatch.config import get_settings
from jobmatch.decisions.base import DecisionModel, Question
from jobmatch.observability.metrics import (
    DECISION_AGREEMENT,
    DECISION_LATENCY,
    DECISIONS,
    SKILL_RESOLUTION,
)
from jobmatch.schemas import ClaimCheck, SkillAssessment, SkillCheck

log = logging.getLogger(__name__)

# --- copied from bench/decisions/build_dataset.py (English variants, used by `jevos@q-en`) ---
SKILL_QUESTION = (
    'Does the CV show that the person has actually used "{s}"? '
    "(Counts as yes: use in a job, internship or concrete project, or a skill listed without "
    "qualification. Does not count: mere 'basic knowledge', a course taken or in progress, "
    "a hobby, or a homonym.)"
)
CLAIM_QUESTION = (
    'Here is a sentence from this person\'s cover letter: "{c}"\n'
    "Is this claim fully supported by the CV (no invented or exaggerated fact, number, "
    "duration, role or collaboration)?"
)

# --- temperature scaling fitted on the calib split by the benchmark:
# bench/decisions/results/summary.json["jevos@q-en"]["metrics"][task]["temperature"]
# (p_k ∝ p_k^(1/T); T < 1 means jevos is slightly UNDER-confident, so it sharpens).
# These temperatures are JEVOS-ONLY: they were fitted on jevos' probabilities. Any other
# backend (DECISIONS_BACKEND=laya / qwen-logits ...) gets T = 1.0 (raw probabilities, no
# calibration) and a one-time warning; see `calibration_temperature`.
TEMPERATURE = {
    "skill_evidence": 0.7351310500673817,
    "claim_supported": 0.8860404832876525,
}

# jevos rejects states > 8192 tokens (no truncation). We skip instead of truncating (a cut
# CV would produce false "no evidence" answers). ~3 chars/token is a conservative bound
# (measured ~4.5 on a French CV, BACKENDS.md); an HTTP 422 from jevos is also a "skip".
MAX_STATE_CHARS = 24_000

LLM_CREDIT = {"full": 1.0, "partial": 0.5, "none": 0.0}  # same rubric as score_match
CONFLICT_CREDIT = 0.5  # strong LLM / decision-model disagreement -> partial credit
CALIBRATED_BACKEND = "jevos"  # the backend TEMPERATURE was fitted for

# --- backend cache (one instance per event loop: httpx clients are loop-bound) --------------
_model: DecisionModel | None = None
_model_loop: asyncio.AbstractEventLoop | None = None
_semaphore: asyncio.Semaphore | None = None
_warned = False
_warned_uncalibrated = False


class DecisionsUnavailable(RuntimeError):
    """The decision layer is disabled or the state cannot be sent (too long)."""


def apply_temperature(p_yes: float, temperature: float) -> float:
    """Calibrated P(yes) for a 2-way decision (same formula as bench/decisions/metrics.py)."""
    eps = 1e-9
    p = min(max(p_yes, eps), 1 - eps)
    logit = (math.log(p) - math.log(1 - p)) / temperature
    return 1 / (1 + math.exp(-logit))


def calibration_temperature(task: str) -> float:
    """Benchmark temperature for jevos; 1.0 (no calibration, warned once) for other backends."""
    global _warned_uncalibrated
    from jobmatch.decisions.registry import canonical_name

    backend = canonical_name(get_settings().decisions_backend)
    if backend == CALIBRATED_BACKEND:
        return TEMPERATURE[task]
    if not _warned_uncalibrated:
        _warned_uncalibrated = True
        log.warning(
            "decisions: no calibration temperature for backend %r (fitted for %r only); "
            "using raw probabilities (T=1.0). Thresholds may be miscalibrated.",
            backend,
            CALIBRATED_BACKEND,
        )
    return 1.0


def get_decision_model() -> DecisionModel | None:
    """The configured backend for the running event loop, or None if disabled."""
    global _model, _model_loop, _semaphore
    settings = get_settings()
    if not settings.decisions_enabled:
        return None
    loop = asyncio.get_running_loop()
    if _model is None or _model_loop is not loop:
        from jobmatch.decisions.registry import canonical_name, get_backend

        name = canonical_name(settings.decisions_backend)
        kwargs: dict = {}
        if name == "jevos":
            kwargs = {"base_url": settings.jevos_url, "timeout": settings.decisions_timeout_s}
        _model = get_backend(name, **kwargs)
        _model_loop = loop
        _semaphore = asyncio.Semaphore(settings.decisions_concurrency)
    return _model


def set_decision_model(model: DecisionModel | None) -> None:
    """Override the backend for the current loop (tests); None resets the cache."""
    global _model, _model_loop, _semaphore
    _model = model
    _model_loop = asyncio.get_running_loop() if model is not None else None
    _semaphore = asyncio.Semaphore(get_settings().decisions_concurrency) if model else None


def _degrade(task: str, exc: BaseException) -> None:
    global _warned
    skipped = isinstance(exc, DecisionsUnavailable)
    DECISIONS.labels(task=task, outcome="skipped" if skipped else "error").inc()
    if skipped:
        log.info("decisions[%s] skipped: %s", task, exc)
    elif not _warned:
        _warned = True
        log.warning(
            "decisions[%s] unavailable, falling back to LLM-only results (%s: %s). "
            "Further failures are logged at DEBUG level.",
            task,
            type(exc).__name__,
            exc,
        )
    else:
        log.debug("decisions[%s] failed: %r", task, exc)


async def _ask_yes(task: str, state: str, questions: Sequence[str]) -> list[float]:
    """Calibrated P(yes) per question; raises on any failure (all or nothing)."""
    model = get_decision_model()
    if model is None:
        raise DecisionsUnavailable("decision layer disabled")
    if len(state) > MAX_STATE_CHARS:
        raise DecisionsUnavailable(f"state too long ({len(state)} > {MAX_STATE_CHARS} chars)")
    sem = _semaphore or asyncio.Semaphore(1)
    temperature = calibration_temperature(task)

    async def one(q: str) -> float:
        async with sem:
            t0 = time.perf_counter()
            try:
                decision = await model.decide(state, Question(kind="yes_no", question=q))
            except RuntimeError as exc:
                if "HTTP 422" in str(exc):  # jevos: state over its 8192-token context
                    raise DecisionsUnavailable(f"rejected by the backend: {exc}") from exc
                raise
            DECISION_LATENCY.labels(task=task).observe(time.perf_counter() - t0)
        return apply_temperature(decision.probs["yes"], temperature)

    probs = await asyncio.gather(*(one(q) for q in questions))
    for p in probs:
        DECISIONS.labels(task=task, outcome="yes" if p >= 0.5 else "no").inc()
    return list(probs)


def fuse_skill(assessment: SkillAssessment, p_evidence: float) -> SkillCheck:
    """Fusion v2: the LLM rating is kept unless it is unsure or strongly contradicted.

    llm_credit c in {1, 0.5, 0}, calibrated p = P(the CV shows the skill):
      - c == 0.5 (partial / unsure)       -> final = p    ("resolved_partial": System 1 decides)
      - c == 1 and p < conflict_low (0.2) -> final = 0.5  ("conflict")
      - c == 0 and p > conflict_high (0.8) -> final = 0.5 ("conflict")
      - otherwise                         -> final = c    ("llm": no systematic bias)
    v1 averaged 0.5 * c + 0.5 * p, which biased scores DOWN (p rarely reaches 1.0, so strong
    profiles lost 5-15 points in the A/B eval) while jevos agreed with the LLM on 96% of skills.
    """
    settings = get_settings()
    llm_credit = LLM_CREDIT[assessment.match]
    if llm_credit == LLM_CREDIT["partial"]:
        final, resolution = p_evidence, "resolved_partial"
    elif (llm_credit == 1.0 and p_evidence < settings.decision_conflict_low) or (
        llm_credit == 0.0 and p_evidence > settings.decision_conflict_high
    ):
        final, resolution = CONFLICT_CREDIT, "conflict"
    else:
        final, resolution = llm_credit, "llm"
    return SkillCheck(
        skill=assessment.skill,
        level=assessment.level,
        llm_credit=llm_credit,
        p_evidence=round(p_evidence, 4),
        final_credit=round(final, 4),
        agree=(llm_credit >= 0.5) == (p_evidence >= 0.5),
        resolution=resolution,
    )


async def check_skill_evidence(
    cv_text: str, assessments: Sequence[SkillAssessment]
) -> list[SkillCheck]:
    """For every LLM-rated offer skill: P(CV shows it) and the fused credit. [] on failure."""
    task = "skill_evidence"
    if not assessments or not get_settings().decisions_enabled:
        return []
    try:
        probs = await _ask_yes(
            task, cv_text, [SKILL_QUESTION.format(s=a.skill) for a in assessments]
        )
    except Exception as exc:  # noqa: BLE001 - degrade, never fail the analysis
        _degrade(task, exc)
        return []
    checks = [fuse_skill(a, p) for a, p in zip(assessments, probs, strict=True)]
    for c in checks:
        DECISION_AGREEMENT.labels(task=task, agree=str(c.agree).lower()).inc()
        SKILL_RESOLUTION.labels(resolution=c.resolution).inc()
    return checks


async def check_claims(cv_text: str, claims: Sequence[str]) -> list[ClaimCheck]:
    """For every claim the reviewer flagged: P(CV supports it); dismissed if >= threshold."""
    task = "claim_supported"
    if not claims or not get_settings().decisions_enabled:
        return []
    try:
        probs = await _ask_yes(task, cv_text, [CLAIM_QUESTION.format(c=c) for c in claims])
    except Exception as exc:  # noqa: BLE001 - degrade, never fail the analysis
        _degrade(task, exc)
        return []
    threshold = get_settings().claim_dismiss_threshold
    checks = [
        ClaimCheck(claim=c, p_supported=round(p, 4), dismissed=p >= threshold)
        for c, p in zip(claims, probs, strict=True)
    ]
    for c in checks:
        # The reviewer said "unsupported": the decision model agrees if P(supported) < 0.5.
        DECISION_AGREEMENT.labels(task=task, agree=str(c.p_supported < 0.5).lower()).inc()
    return checks
