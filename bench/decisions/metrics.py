"""Metric functions for the decision benchmark (pure Python, no numpy).

Every prediction is a dict label -> probability (sums to 1) plus a gold label.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Probs = dict[str, float]
EPS = 1e-12


def argmax(p: Probs) -> str:
    return max(p, key=p.__getitem__)


def accuracy(preds: Sequence[Probs], gold: Sequence[str]) -> float:
    return sum(argmax(p) == y for p, y in zip(preds, gold, strict=True)) / len(gold)


def macro_f1(preds: Sequence[Probs], gold: Sequence[str], labels: Sequence[str]) -> float:
    """Unweighted mean of per-class F1 over `labels` (classes absent from gold AND
    predictions are skipped so they do not count as 0)."""
    hats = [argmax(p) for p in preds]
    f1s = []
    for c in labels:
        tp = sum(h == c and y == c for h, y in zip(hats, gold, strict=True))
        fp = sum(h == c and y != c for h, y in zip(hats, gold, strict=True))
        fn = sum(h != c and y == c for h, y in zip(hats, gold, strict=True))
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return sum(f1s) / len(f1s) if f1s else float("nan")


def brier(preds: Sequence[Probs], gold: Sequence[str]) -> float:
    """Multi-class Brier score: mean over items of sum_k (p_k - 1[k=y])^2. Range [0, 2]."""
    tot = 0.0
    for p, y in zip(preds, gold, strict=True):
        tot += sum((v - (1.0 if k == y else 0.0)) ** 2 for k, v in p.items())
    return tot / len(gold)


def nll(preds: Sequence[Probs], gold: Sequence[str]) -> float:
    return -sum(math.log(max(p.get(y, 0.0), EPS)) for p, y in zip(preds, gold, strict=True)) / len(
        gold
    )


def ece(preds: Sequence[Probs], gold: Sequence[str], n_bins: int = 10) -> float:
    """Expected calibration error of the top-label confidence, equal-width bins on [0, 1]."""
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for p, y in zip(preds, gold, strict=True):
        c = max(p.values())
        idx = min(int(c * n_bins), n_bins - 1)
        bins[idx].append((c, argmax(p) == y))
    n = len(gold)
    total = 0.0
    for b in bins:
        if b:
            conf = sum(c for c, _ in b) / len(b)
            acc = sum(ok for _, ok in b) / len(b)
            total += len(b) / n * abs(conf - acc)
    return total


def apply_temperature(p: Probs, t: float) -> Probs:
    """p_k^(1/T) renormalised == softmax(log p / T)."""
    logs = {k: math.log(max(v, EPS)) / t for k, v in p.items()}
    m = max(logs.values())
    exps = {k: math.exp(v - m) for k, v in logs.items()}
    z = sum(exps.values())
    return {k: v / z for k, v in exps.items()}


def fit_temperature(
    preds: Sequence[Probs], gold: Sequence[str], lo: float = 0.05, hi: float = 20.0
) -> float:
    """Temperature minimising NLL on (preds, gold); golden-section search on log T."""
    if not preds:
        return 1.0

    def loss(log_t: float) -> float:
        t = math.exp(log_t)
        return nll([apply_temperature(p, t) for p in preds], gold)

    a, b = math.log(lo), math.log(hi)
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = loss(c), loss(d)
    for _ in range(80):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = loss(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = loss(d)
    return math.exp((a + b) / 2)


def automatable_share(preds: Sequence[Probs], gold: Sequence[str], budget: float = 0.05) -> float:
    """Largest coverage (fraction of items answered, most confident first) such that the
    error rate among answered items is <= budget. Cuts only between distinct confidences
    (ties are answered together). Oracle threshold on the evaluated set."""
    rows = sorted(
        ((max(p.values()), argmax(p) == y) for p, y in zip(preds, gold, strict=True)),
        key=lambda r: -r[0],
    )
    n = len(rows)
    best = 0
    errors = 0
    for i, (conf, ok) in enumerate(rows):
        errors += not ok
        k = i + 1
        boundary = k == n or rows[k][0] < conf
        if boundary and errors / k <= budget:
            best = k
    return best / n if n else float("nan")


def percentile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    pos = (len(s) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def wilson_low(p_hat: float, n: int, z: float = 1.96) -> float:
    """Lower bound of the Wilson score interval for a proportion (95% by default)."""
    if n == 0:
        return float("nan")
    denom = 1 + z * z / n
    centre = p_hat + z * z / (2 * n)
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    return (centre - margin) / denom
