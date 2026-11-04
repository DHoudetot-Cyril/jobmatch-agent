"""Pure aggregation helpers shared by `run_eval.py` and `compare.py` (no LLM, no DeepEval).

A *label directory* (`evals/results/<label>/` or the committed `evals/baseline/`) contains:

    outputs/<case>.json          one cached graph output per case (normal run)
    outputs/<case>__r<i>.json    repeated graph runs (`--repeat N`), i = 0..N-1
    summary.json                 judged metrics + per-case run stats (optional for compare)
    report.md

Everything here works on outputs produced before the decision layer existed (the baseline):
new fields (`match.skill_checks`, `review.unsupported_claims`, `review.claim_checks`,
`trace`) are optional and their stats are `None` when absent.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any

METRIC_ORDER = [
    "score_in_range",
    "skills_recall",
    "gaps_recall",
    "letter_faithfulness",
    "letter_relevance",
    "gap_usefulness",
]

DECISION_NODES = ("verify_skills", "verify_claims")

_REPEAT_RE = re.compile(r"^(?P<case>.+)__r(?P<i>\d+)$")


# --------------------------------------------------------------------------- outputs on disk


def load_label_outputs(label_dir: Path) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    """Return (main output per case, repeated outputs per case sorted by repeat index).

    The main output of a case is `<case>.json`, or `<case>__r0.json` when only repeats exist.
    """
    singles: dict[str, dict] = {}
    repeats: dict[str, list[tuple[int, dict]]] = {}
    out_dir = label_dir / "outputs"
    if out_dir.is_dir():
        for path in sorted(out_dir.glob("*.json")):
            data = json.loads(path.read_text())
            m = _REPEAT_RE.match(path.stem)
            if m:
                repeats.setdefault(m["case"], []).append((int(m["i"]), data))
            else:
                singles[path.stem] = data
    sorted_repeats = {
        c: [d for _, d in sorted(runs, key=lambda t: t[0])] for c, runs in repeats.items()
    }
    main = dict(singles)
    for case, runs in sorted_repeats.items():
        main.setdefault(case, runs[0])
    return main, sorted_repeats


def load_summary(label_dir: Path) -> dict | None:
    path = label_dir / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


# --------------------------------------------------------------------------- per-run stats


def decision_stats(output: dict) -> dict[str, Any] | None:
    """Decision-layer stats of one graph output, or None if the output has no such fields."""
    match = output.get("match") or {}
    review = output.get("review") or {}
    trace = output.get("trace") or []
    has_fields = any(k in match for k in ("skill_checks",)) or any(
        k in review for k in ("unsupported_claims", "claim_checks")
    )
    has_nodes = any(n in trace for n in DECISION_NODES)
    if not has_fields and not has_nodes:
        return None
    skill_checks = match.get("skill_checks") or []
    claim_checks = review.get("claim_checks") or []
    agrees = [bool(c.get("agree")) for c in skill_checks if c.get("agree") is not None]
    overrides = [
        c
        for c in skill_checks
        if c.get("final_credit") is not None
        and c.get("llm_credit") is not None
        and c["final_credit"] != c["llm_credit"]
    ]
    return {
        "skill_checks": len(skill_checks),
        "skill_agreement": round(sum(agrees) / len(agrees), 3) if agrees else None,
        "skill_overrides": len(overrides),
        "claims_flagged": len(review.get("unsupported_claims") or []),
        "claims_checked": len(claim_checks),
        "claims_dismissed": sum(1 for c in claim_checks if c.get("dismissed")),
        "decision_count": len(skill_checks) + len(claim_checks),
        "verify_nodes_in_trace": [n for n in trace if n in DECISION_NODES],
    }


def run_stats(output: dict) -> dict[str, Any]:
    """Operational stats of one graph output (works for baseline and new outputs)."""
    if "error" in output:
        return {"error": output["error"]}
    review = output.get("review") or {}
    return {
        "score": (output.get("match") or {}).get("score"),
        "graph_seconds": output.get("elapsed_s"),
        "review_iterations": output.get("review_iterations"),
        "reviewer_approved": review.get("approved"),
        "decisions_enabled": output.get("decisions_enabled"),
        "decisions": decision_stats(output),
    }


def score_stability(outputs: list[dict]) -> dict[str, Any] | None:
    """mean / sample std / min / max of `match.score` over repeated runs (None if no score)."""
    scores = [
        o["match"]["score"]
        for o in outputs
        if "error" not in o and (o.get("match") or {}).get("score") is not None
    ]
    if not scores:
        return None
    return {
        "n": len(scores),
        "scores": scores,
        "mean": round(statistics.fmean(scores), 2),
        "std": round(statistics.stdev(scores), 2) if len(scores) > 1 else None,
        "min": min(scores),
        "max": max(scores),
        "failed_runs": len(outputs) - len(scores),
    }


# --------------------------------------------------------------------------- summaries


def metric_averages(results: dict[str, dict], cases: list[str] | None = None) -> dict:
    """Average of every metric over `cases` (default: all), ignoring missing values."""
    ids = cases if cases is not None else list(results)
    out: dict[str, float | None] = {}
    for name in METRIC_ORDER:
        vals = [
            v
            for cid in ids
            if cid in results
            and (v := ((results[cid].get("metrics") or {}).get(name) or {}).get("value"))
            is not None
        ]
        out[name] = round(sum(vals) / len(vals), 3) if vals else None
    return out


def failed_thresholds(avgs: dict[str, float | None], thresholds: dict[str, float]) -> list[str]:
    return [n for n, thr in thresholds.items() if avgs.get(n) is None or avgs[n] < thr]


def merge_summaries(existing: dict | None, new: dict, case_order: list[str]) -> dict:
    """Merge a (possibly partial, `--cases`) run into the label's existing summary.

    Cases of `new` replace the same cases of `existing`; other cases are kept. Averages and
    failed thresholds are recomputed over the merged case set, `runs` keeps the history.
    Raises ValueError when the two runs used a different decision-layer config.
    """
    history = [] if existing is None else existing.get("runs", [_run_entry(existing)])
    existing = existing or {}
    old_cfg = (existing.get("config") or {}).get("decisions")
    new_cfg = (new.get("config") or {}).get("decisions")
    if old_cfg is not None and new_cfg is not None and old_cfg != new_cfg:
        raise ValueError(
            f"label already holds a run with decisions={old_cfg}, this run is decisions="
            f"{new_cfg}; use another --label"
        )
    results = {**existing.get("results", {}), **new.get("results", {})}
    order = {cid: i for i, cid in enumerate(case_order)}
    cases = sorted(results, key=lambda c: (order.get(c, len(order)), c))
    thresholds = new.get("thresholds") or existing.get("thresholds") or {}
    avgs = metric_averages(results, cases)
    merged = {
        **existing,
        **new,
        "cases": cases,
        "results": results,
        "averages": avgs,
        "thresholds": thresholds,
        "failed_thresholds": failed_thresholds(avgs, thresholds),
        "runs": [*history, _run_entry(new)],
    }
    merged["config"] = new.get("config") or existing.get("config")
    return merged


def _run_entry(summary: dict) -> dict:
    return {
        "generated_at": summary.get("generated_at"),
        "cases": summary.get("cases"),
        "config": summary.get("config"),
    }
