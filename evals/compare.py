"""A/B comparison of two eval label directories (no LLM, no GPU).

    uv run --group eval python evals/compare.py A_DIR B_DIR [--out FILE]

A_DIR / B_DIR are label directories: `evals/baseline` (committed snapshot, Qwen only) or
`evals/results/<label>` (written by `run_eval.py --label <label>`). Judged metrics come from
each `summary.json`; operational stats (graph time, review iterations, approval, decision
layer) are derived from `outputs/*.json`, so a snapshot without those summary fields still
works ("n/a" when even the outputs lack them). Averages are computed over the cases present
in BOTH directories. Default output: `evals/results/compare_<A>_vs_<B>.md` (also printed).
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # `python evals/compare.py`: make `evals` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from evals.aggregate import (  # noqa: E402
    METRIC_ORDER,
    load_label_outputs,
    load_summary,
    run_stats,
    score_stability,
)

EVALS_DIR = Path(__file__).parent
DATASET_DIR = EVALS_DIR / "dataset"
NA = "n/a"


# --------------------------------------------------------------------------- loading


def load_side(label_dir: Path) -> dict[str, Any]:
    """Everything compare needs about one label directory."""
    summary = load_summary(label_dir) or {}
    main, repeats = load_label_outputs(label_dir)
    cases = list(summary.get("results") or {}) or sorted(main)
    for cid in main:
        if cid not in cases:
            cases.append(cid)
    results = summary.get("results") or {}
    runs = {}
    for cid in cases:
        if cid in main:
            runs[cid] = run_stats(main[cid])
        else:  # no output on disk: fall back to what the summary recorded
            runs[cid] = (results.get(cid) or {}).get("run") or {"score": results[cid].get("score")}
    stability = {cid: score_stability(r) for cid, r in repeats.items() if len(r) > 1}
    config = summary.get("config") or {}
    return {
        "name": label_dir.resolve().name,
        "dir": str(label_dir),
        "config": config,
        "cases": cases,
        "metrics": {cid: (results.get(cid) or {}).get("metrics") or {} for cid in cases},
        "runs": runs,
        "stability": stability,
    }


def expected_ranges() -> dict[str, tuple[int, int]]:
    out = {}
    for p in sorted(DATASET_DIR.glob("*.yaml")):
        data = yaml.safe_load(p.read_text())
        out[data["id"]] = tuple(data["expected_score_range"])
    return out


# --------------------------------------------------------------------------- math / format


def mean(values: list[float | int | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return statistics.fmean(vals) if vals else None


def rate(values: list[bool | None]) -> float | None:
    vals = [bool(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def delta(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else b - a


def arrow(d: float | None, higher_is_better: bool = True, eps: float = 1e-9) -> str:
    """↑ / ↓ show the direction of B vs A; (better)/(worse) depends on the metric."""
    if d is None:
        return ""
    if abs(d) < eps:
        return "="
    up = d > 0
    good = up == higher_is_better
    return f"{'↑' if up else '↓'} {'better' if good else 'worse'}"


def fmt(v: float | None, digits: int = 2, signed: bool = False) -> str:
    if v is None:
        return NA
    if signed and abs(v) < 10 ** (-digits) / 2:
        return f"{0:.{digits}f}"
    return f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"


def fmt_pct(v: float | None) -> str:
    return NA if v is None else f"{100 * v:.0f}%"


# --------------------------------------------------------------------------- aggregates


def metric_avg(side: dict, name: str, cases: list[str]) -> float | None:
    return mean([(side["metrics"][c].get(name) or {}).get("value") for c in cases])


def ops(side: dict, cases: list[str]) -> dict[str, float | None]:
    runs = [side["runs"][c] for c in cases]
    ok = [r for r in runs if not r.get("error")]
    return {
        "approval": rate([r.get("reviewer_approved") for r in ok]),
        "iterations": mean([r.get("review_iterations") for r in ok]),
        "seconds": mean([r.get("graph_seconds") for r in ok]),
        "errors": float(len(runs) - len(ok)),
    }


def decision_totals(side: dict, cases: list[str]) -> dict[str, Any] | None:
    decs = [d for c in cases if (d := side["runs"][c].get("decisions"))]
    if not decs:
        return None
    checks = sum(d["skill_checks"] for d in decs)
    # agreement weighted by the number of skill checks of each case
    weighted = [
        (d["skill_agreement"], d["skill_checks"]) for d in decs if d["skill_agreement"] is not None
    ]
    n_w = sum(w for _, w in weighted)
    return {
        "cases": len(decs),
        "skill_checks": checks,
        "skill_agreement": sum(a * w for a, w in weighted) / n_w if n_w else None,
        "skill_overrides": sum(d["skill_overrides"] for d in decs),
        "claims_flagged": sum(d["claims_flagged"] for d in decs),
        "claims_checked": sum(d["claims_checked"] for d in decs),
        "claims_dismissed": sum(d["claims_dismissed"] for d in decs),
        "decision_count": sum(d["decision_count"] for d in decs),
    }


# --------------------------------------------------------------------------- report


def _cfg(side: dict) -> str:
    cfg = side["config"]
    dec = cfg.get("decisions")
    if dec is None:  # older snapshot: infer from the outputs
        dec = "on" if any(r.get("decisions") for r in side["runs"].values()) else "off (baseline)"
    return f"`{side['dir']}` (decisions: {dec})"


def build_report(a: dict, b: dict, ranges: dict[str, tuple[int, int]] | None = None) -> str:
    ranges = ranges if ranges is not None else {}
    common = [c for c in a["cases"] if c in b["cases"]]
    only_a = [c for c in a["cases"] if c not in b["cases"]]
    only_b = [c for c in b["cases"] if c not in a["cases"]]
    A, B = a["name"], b["name"]
    lines = [
        f"# Eval comparison: {A} vs {B}",
        "",
        f"- **A** = {_cfg(a)}",
        f"- **B** = {_cfg(b)}",
        f"- Cases compared: {len(common)} (present in both)"
        + (f"; only in A: {', '.join(only_a)}" if only_a else "")
        + (f"; only in B: {', '.join(only_b)}" if only_b else ""),
        "- Δ = B − A. ↑/↓ = direction of the change, better/worse = whether it is an "
        "improvement for that metric.",
        "",
        "## Judged metrics (average over cases, 0-1, higher is better)",
        "",
        f"| metric | A ({A}) | B ({B}) | Δ | |",
        "|---|---|---|---|---|",
    ]
    for name in METRIC_ORDER:
        va, vb = metric_avg(a, name, common), metric_avg(b, name, common)
        d = delta(va, vb)
        lines.append(f"| `{name}` | {fmt(va, 3)} | {fmt(vb, 3)} | {fmt(d, 3, True)} | {arrow(d)} |")

    lines += [
        "",
        "## Match score per case",
        "",
        f"| case | expected | A ({A}) | B ({B}) | Δ |",
        "|---|---|---|---|---|",
    ]
    for c in common:
        sa, sb = a["runs"][c].get("score"), b["runs"][c].get("score")
        lo_hi = ranges.get(c)
        exp = f"{lo_hi[0]}-{lo_hi[1]}" if lo_hi else NA

        def cell(s, lo_hi=lo_hi):
            if s is None:
                return NA
            if lo_hi and not lo_hi[0] <= s <= lo_hi[1]:
                return f"{s} (out of range)"
            return str(s)

        d = delta(sa, sb)
        lines.append(f"| {c} | {exp} | {cell(sa)} | {cell(sb)} | {fmt(d, 0, True)} |")

    oa, ob = ops(a, common), ops(b, common)
    rows = [
        ("reviewer approval rate", oa["approval"], ob["approval"], True, "pct"),
        ("avg review iterations", oa["iterations"], ob["iterations"], False, 2),
        ("avg graph seconds per case", oa["seconds"], ob["seconds"], False, 0),
        ("graph errors", oa["errors"], ob["errors"], False, 0),
    ]
    lines += [
        "",
        "## Graph behaviour",
        "",
        f"| | A ({A}) | B ({B}) | Δ | |",
        "|---|---|---|---|---|",
    ]
    for label, va, vb, hib, digits in rows:
        d = delta(va, vb)
        if digits == "pct":
            pts = None if d is None else 100 * d
            cells = f"{fmt_pct(va)} | {fmt_pct(vb)} | {fmt(pts, 0, True)}"
            cells += " pts" if d is not None else ""
        else:
            cells = f"{fmt(va, digits)} | {fmt(vb, digits)} | {fmt(d, digits, True)}"
        lines.append(f"| {label} | {cells} | {arrow(d, hib)} |")

    lines += ["", "## Score stability (repeated graph runs)", ""]
    stab_cases = [c for c in common if c in a["stability"] or c in b["stability"]]
    if not stab_cases:
        lines.append(
            "No repeated runs in either directory (`run_eval.py --repeat N` to measure it)."
        )
    else:
        lines += [
            f"| case | A ({A}) mean ± std [min-max] (n) | B ({B}) mean ± std [min-max] (n) "
            "| Δ std |",
            "|---|---|---|---|",
        ]
        for c in stab_cases:
            sa, sb = a["stability"].get(c), b["stability"].get(c)
            d = delta(sa and sa["std"], sb and sb["std"])
            lines.append(
                f"| {c} | {_stab(sa)} | {_stab(sb)} | {fmt(d, 2, True)} {arrow(d, False)} |"
            )

    lines += ["", "## Decision layer (jevos)", ""]
    any_dec = False
    for side, letter in ((a, "A"), (b, "B")):
        tot = decision_totals(side, common)
        if tot is None:
            lines.append(f"- **{letter}** ({side['name']}): no decision-layer data (n/a).")
            continue
        any_dec = True
        lines.append(
            f"- **{letter}** ({side['name']}): {tot['decision_count']} jevos decisions on "
            f"{tot['cases']} cases — {tot['skill_checks']} skill checks, LLM/jevos agreement "
            f"{fmt_pct(tot['skill_agreement'])}, {tot['skill_overrides']} credits overridden; "
            f"{tot['claims_checked']} reviewer claims checked, {tot['claims_flagged']} flagged "
            f"unsupported, {tot['claims_dismissed']} dismissed."
        )
    if any_dec:
        lines += [
            "",
            "| case | side | skill checks | agreement | overrides | claims flagged "
            "| claims dismissed | decisions |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for c in common:
            for side, letter in ((a, "A"), (b, "B")):
                d = side["runs"][c].get("decisions")
                if not d:
                    continue
                lines.append(
                    f"| {c} | {letter} | {d['skill_checks']} | {fmt_pct(d['skill_agreement'])} "
                    f"| {d['skill_overrides']} | {d['claims_flagged']} | {d['claims_dismissed']} "
                    f"| {d['decision_count']} |"
                )
    lines.append("")
    return "\n".join(lines)


def _stab(s: dict | None) -> str:
    if not s:
        return NA
    std = NA if s["std"] is None else f"{s['std']:.2f}"
    return f"{s['mean']:.1f} ± {std} [{s['min']}-{s['max']}] ({s['n']})"


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("a_dir", type=Path, help="label directory A (e.g. evals/baseline)")
    parser.add_argument("b_dir", type=Path, help="label directory B (e.g. evals/results/jevos)")
    parser.add_argument("--out", type=Path, help="markdown file (default: evals/results/...)")
    args = parser.parse_args(argv)
    for d in (args.a_dir, args.b_dir):
        if not d.is_dir():
            parser.error(f"not a directory: {d}")
    a, b = load_side(args.a_dir), load_side(args.b_dir)
    report = build_report(a, b, expected_ranges())
    out = args.out or EVALS_DIR / "results" / f"compare_{a['name']}_vs_{b['name']}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(report)
    print(f"written: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
