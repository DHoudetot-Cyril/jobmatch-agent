"""Evaluation runner: graph run (cached) -> metrics -> report + Langfuse scores.

    uv run --group eval python -m evals.run_eval [--label NAME] [--cases id1,id2]
        [--decisions on|off] [--repeat N [--judge-all]] [--refresh] [--no-langfuse]

1. For each dataset case, run the real graph (MemorySaver) up to the human_review interrupt,
   at most `GRAPH_CONCURRENCY` runs at a time (llama-server has 2 slots). Outputs are cached
   in `evals/results/<label>/outputs/<case>.json` (or `<case>__r<i>.json` with `--repeat`)
   and reused unless `--refresh`.
2. Compute deterministic metrics + G-Eval metrics (local Qwen judge). With `--repeat`, only
   run r0 is judged unless `--judge-all` (then every metric is the mean over the repeats).
3. Write `evals/results/<label>/report.md` and `summary.json`, post every metric as a
   Langfuse score on the case's trace (skipped silently if Langfuse is down / --no-langfuse).
   A `--cases` run is MERGED into the label's existing summary (other cases are kept, the
   averages are recomputed over the merged set) — it never overwrites a full report.
4. Exit 1 if an average falls below `THRESHOLDS`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # `python evals/run_eval.py`: make `evals` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402
import yaml  # noqa: E402

from evals.aggregate import (  # noqa: E402
    METRIC_ORDER,
    load_label_outputs,
    load_summary,
    merge_summaries,
    run_stats,
    score_stability,
)

# Averages below these values make the run fail (exit code 1).
THRESHOLDS: dict[str, float] = {
    "letter_faithfulness": 0.7,
    "score_in_range": 0.6,
}
GRAPH_CONCURRENCY = 2
JUDGE_CONCURRENCY = 2

EVALS_DIR = Path(__file__).parent
DATASET_DIR = EVALS_DIR / "dataset"
RESULTS_ROOT = EVALS_DIR / "results"
DEFAULT_LABEL = "latest"

logger = logging.getLogger("evals")


def label_dir(label: str) -> Path:
    return RESULTS_ROOT / label


def output_path(out_dir: Path, case_id: str, repeat: int | None) -> Path:
    return out_dir / (f"{case_id}.json" if repeat is None else f"{case_id}__r{repeat}.json")


# --------------------------------------------------------------------------- dataset


def load_cases(selected: list[str] | None = None) -> list[dict]:
    cases = [yaml.safe_load(p.read_text()) for p in sorted(DATASET_DIR.glob("*.yaml"))]
    if selected:
        unknown = set(selected) - {c["id"] for c in cases}
        if unknown:
            raise SystemExit(f"unknown case id(s): {sorted(unknown)}")
        cases = [c for c in cases if c["id"] in selected]
    return cases


# --------------------------------------------------------------------------- graph


def _dump(value: Any) -> Any:
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


async def run_case(case: dict, use_langfuse: bool, run_tag: str = "") -> dict:
    """Run the graph up to the human_review interrupt and return a JSON-able output."""
    from langgraph.checkpoint.memory import MemorySaver

    from jobmatch.config import get_settings
    from jobmatch.graph.build import build_graph
    from jobmatch.observability.tracing import get_langfuse_handler, langfuse_run_config

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    # One trace per (case, graph run); the seed is cached so scores land on the right trace.
    trace_seed = f"eval-{case['id']}{run_tag}-{stamp}"
    analysis_id = f"eval-{case['id']}{run_tag}-{uuid.uuid4().hex[:8]}"
    handler = get_langfuse_handler(trace_seed=trace_seed) if use_langfuse else None
    config: dict[str, Any] = {
        "configurable": {"thread_id": analysis_id},
        **langfuse_run_config(analysis_id),
    }
    config["tags"] = [*config["tags"], "eval"]
    config["metadata"]["langfuse_tags"] = [*config["metadata"]["langfuse_tags"], "eval"]
    if handler is not None:
        config["callbacks"] = [handler]

    graph = build_graph(checkpointer=MemorySaver())
    start = time.perf_counter()
    await graph.ainvoke({"offer_text": case["offer_text"], "cv_text": case["cv_text"]}, config)
    elapsed = time.perf_counter() - start
    snapshot = await graph.aget_state(config)
    if not snapshot.interrupts:
        raise RuntimeError(f"{case['id']}: graph did not reach human_review")
    values = snapshot.values
    trace_id = None
    if handler is not None:
        from langfuse import Langfuse

        trace_id = Langfuse.create_trace_id(seed=trace_seed)
    return {
        "case_id": case["id"],
        "analysis_id": analysis_id,
        "trace_seed": trace_seed,
        "trace_id": trace_id,
        "created_at": stamp,
        "elapsed_s": round(elapsed, 1),
        # None when the setting does not exist (graph older than the decision layer)
        "decisions_enabled": getattr(get_settings(), "decisions_enabled", None),
        "offer": _dump(values.get("offer")),
        "profile": _dump(values.get("profile")),
        "match": _dump(values.get("match")),
        "cover_letter": _dump(values.get("cover_letter")),
        "review": _dump(values.get("review")),
        "review_iterations": values.get("review_iterations"),
        "trace": list(values.get("trace") or []),
        "status": str(values.get("status")),
    }


async def get_outputs(
    cases: list[dict], out_dir: Path, refresh: bool, use_langfuse: bool, repeat: int | None
) -> dict[str, list[dict]]:
    """Graph outputs per case: one item, or `repeat` items (r0..rN-1) with `--repeat`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(GRAPH_CONCURRENCY)
    indices: list[int | None] = [None] if repeat is None else list(range(repeat))
    outputs: dict[tuple[str, int | None], dict] = {}

    async def one(case: dict, idx: int | None) -> None:
        key, name = (case["id"], idx), case["id"] + ("" if idx is None else f"__r{idx}")
        path = output_path(out_dir, case["id"], idx)
        if path.exists() and not refresh:
            outputs[key] = json.loads(path.read_text())
            logger.info("cache hit: %s", name)
            return
        async with sem:
            logger.info("graph run: %s ...", name)
            try:
                out = await run_case(case, use_langfuse, "" if idx is None else f"-r{idx}")
            except Exception as exc:  # one failing case must not kill the whole run
                logger.exception("graph run failed: %s", name)
                outputs[key] = {"case_id": case["id"], "error": repr(exc)}
                return
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
        outputs[key] = out
        logger.info("graph done: %s in %.0fs", name, out["elapsed_s"])

    await asyncio.gather(*(one(c, i) for c in cases for i in indices))
    if use_langfuse:
        from jobmatch.observability.tracing import flush_langfuse

        flush_langfuse()
    return {c["id"]: [outputs[(c["id"], i)] for i in indices] for c in cases}


# --------------------------------------------------------------------------- judging


async def evaluate_case(case: dict, output: dict, judge, sem: asyncio.Semaphore):
    from evals.metrics import build_geval_metrics, deterministic_metrics, geval_test_cases
    from jobmatch.schemas import MatchResult

    if "error" in output:
        return {name: {"value": None, "reason": output["error"]} for name in METRIC_ORDER}
    results = deterministic_metrics(case, MatchResult.model_validate(output["match"]))
    metrics = build_geval_metrics(judge)
    test_cases = geval_test_cases(case, output)

    async def judge_one(name: str) -> None:
        async with sem:
            start = time.perf_counter()
            try:
                await metrics[name].a_measure(test_cases[name], _show_indicator=False)
                results[name] = {
                    "value": round(float(metrics[name].score), 3),
                    "reason": metrics[name].reason,
                }
            except Exception as exc:
                logger.exception("judge failed: %s/%s", case["id"], name)
                results[name] = {"value": None, "reason": f"judge error: {exc!r}"}
            logger.info(
                "judged %s/%s = %s (%.0fs)",
                case["id"],
                name,
                results[name]["value"],
                time.perf_counter() - start,
            )

    await asyncio.gather(*(judge_one(n) for n in metrics))
    return results


def mean_metrics(per_repeat: list[dict]) -> dict:
    """Per-metric mean over judged repeats (`--judge-all`); the reason is r0's."""
    out = {}
    for name in METRIC_ORDER:
        vals = [r[name]["value"] for r in per_repeat if r[name]["value"] is not None]
        out[name] = {
            "value": round(sum(vals) / len(vals), 3) if vals else None,
            "reason": f"mean of {len(vals)} runs; r0: {per_repeat[0][name].get('reason')}",
            "values": [r[name]["value"] for r in per_repeat],
        }
    return out


# --------------------------------------------------------------------------- Langfuse


def langfuse_score_bodies(outputs: dict[str, dict], results: dict[str, dict]) -> list[dict]:
    """Score payloads for every numeric judge metric of every traced case.

    `results[case]` is a summary entry `{"score": int, "trace_id", "metrics": {name: {"value",
    "reason", ...}}, "run", "stability", ...}` (a bare `{name: metric}` dict is accepted too).
    Anything that is not a numeric metric (ints, aggregates, None values) is skipped.
    """
    bodies = []
    for case_id, entry in results.items():
        trace_id = (outputs.get(case_id) or {}).get("trace_id")
        if not trace_id or not isinstance(entry, dict):
            continue
        metrics = entry.get("metrics", entry)
        if not isinstance(metrics, dict):
            continue
        for name, m in metrics.items():
            if not isinstance(m, dict):
                continue
            value = m.get("value")
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            bodies.append(
                {
                    # deterministic id: re-judging overwrites instead of duplicating
                    "id": f"{trace_id}-{name}",
                    "traceId": trace_id,
                    "name": f"eval_{name}",
                    "value": value,
                    "dataType": "NUMERIC",
                    "comment": str(m.get("reason") or "")[:1000],
                }
            )
    return bodies


def post_langfuse_scores(outputs: dict[str, dict], results: dict[str, dict]) -> int:
    """POST /api/public/scores for every metric. Returns the number of scores posted."""
    from jobmatch.config import get_settings

    s = get_settings()
    auth = (s.langfuse_public_key, s.langfuse_secret_key)
    url = f"{s.langfuse_host.rstrip('/')}/api/public/scores"
    posted = 0
    try:
        with httpx.Client(timeout=5.0, auth=auth) as client:
            for body in langfuse_score_bodies(outputs, results):
                resp = client.post(url, json=body)
                if resp.status_code >= 300:
                    logger.warning("score rejected %s: %s", resp.status_code, resp.text)
                    continue
                posted += 1
    except httpx.HTTPError as exc:
        logger.info("langfuse unreachable, scores skipped (%s)", exc)
    return posted


# --------------------------------------------------------------------------- report


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.2f}"


def write_report(out_path: Path, summary: dict, case_defs: dict[str, dict], outputs: dict) -> None:
    """Render report.md from the (merged) summary and the outputs on disk."""
    cfg = summary.get("config") or {}
    avgs, failures = summary["averages"], summary["failed_thresholds"]
    lines = [
        "# JobMatch evaluation report",
        "",
        f"Generated {summary['generated_at']} · label `{cfg.get('label')}` · decisions "
        f"`{cfg.get('decisions')}` · judge: {cfg.get('judge_model')} (local) · cases: "
        f"{len(summary['cases'])} · last run judged {summary.get('judge_seconds', 0):.0f}s",
        "",
        "| case | score (expected) | in range | skills recall | gaps recall | faithfulness "
        "| relevance | gap usefulness |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cid in summary["cases"]:
        res, case = summary["results"][cid], case_defs.get(cid, {})
        score = res.get("score")
        if (res.get("run") or {}).get("error"):
            score = "ERROR"
        elif score is None:
            score = "—"
        lo, hi = case.get("expected_score_range", ("?", "?"))
        cells = " | ".join(_fmt((res["metrics"].get(n) or {}).get("value")) for n in METRIC_ORDER)
        lines.append(f"| {cid} | {score} ({lo}-{hi}) | {cells} |")
    avg_cells = " | ".join(f"**{_fmt(avgs[n])}**" for n in METRIC_ORDER)
    lines.append(f"| **average** | | {avg_cells} |")
    lines += ["", "## Thresholds", ""]
    for name, thr in summary["thresholds"].items():
        status = "FAIL" if name in failures else "ok"
        lines.append(f"- `{name}` average {_fmt(avgs.get(name))} (>= {thr}): **{status}**")

    stab = {cid: r["stability"] for cid, r in summary["results"].items() if r.get("stability")}
    if any(s["n"] > 1 for s in stab.values()):
        lines += ["", "## Score stability (repeated graph runs)", ""]
        lines += ["| case | n | scores | mean | std | min | max |", "|---|---|---|---|---|---|---|"]
        for cid, s in stab.items():
            lines.append(
                f"| {cid} | {s['n']} | {', '.join(map(str, s['scores']))} | {s['mean']} | "
                f"{'—' if s['std'] is None else s['std']} | {s['min']} | {s['max']} |"
            )

    lines += ["", "## Details", ""]
    for cid in summary["cases"]:
        res, out = summary["results"][cid], outputs.get(cid) or {}
        lines += [f"### {cid}", "", f"_{case_defs.get(cid, {}).get('description', '')}_", ""]
        run = res.get("run") or {}
        if run.get("error") or "error" in out:
            lines += [f"Graph error: `{run.get('error') or out.get('error')}`", ""]
            continue
        lines.append(
            f"- graph run: {run.get('graph_seconds')}s, review iterations "
            f"{run.get('review_iterations')}, reviewer approved: {run.get('reviewer_approved')}"
        )
        if dec := run.get("decisions"):
            lines.append(
                f"- decision layer: {dec['skill_checks']} skill checks (agreement "
                f"{_fmt(dec['skill_agreement'])}, {dec['skill_overrides']} overrides), "
                f"{dec['claims_flagged']} claims flagged, {dec['claims_dismissed']} dismissed"
            )
        if match := out.get("match"):
            lines += [
                f"- matched skills: {', '.join(match['matched_skills']) or '(none)'}",
                f"- gaps: {', '.join(g['skill'] for g in match['gaps']) or '(none)'}",
                f"- letter language: {(out.get('cover_letter') or {}).get('language')}",
            ]
        for name in METRIC_ORDER:
            m = res["metrics"].get(name) or {}
            reason = (m.get("reason") or "").replace("\n", " ")
            lines.append(f"- **{name}** {_fmt(m.get('value'))} — {reason}")
        lines.append("")
    out_path.write_text("\n".join(lines))


# --------------------------------------------------------------------------- main


def apply_decisions_flag(value: str | None) -> None:
    """Set DECISIONS_ENABLED before jobmatch settings are read (get_settings is lru_cached)."""
    if value is None:
        return
    os.environ["DECISIONS_ENABLED"] = "true" if value == "on" else "false"
    from jobmatch.config import get_settings

    get_settings.cache_clear()


async def main_async(args: argparse.Namespace) -> int:
    from evals.judge import QwenJudge
    from jobmatch.config import get_settings

    settings = get_settings()
    selected = [c.strip() for c in args.cases.split(",")] if args.cases else None
    all_cases = load_cases()
    cases = load_cases(selected)
    ldir = label_dir(args.label)
    out_dir = ldir / "outputs"
    use_langfuse = not args.no_langfuse and settings.langfuse_enabled
    effective = getattr(settings, "decisions_enabled", None)
    decisions = args.decisions or ("n/a" if effective is None else ("on" if effective else "off"))
    if args.decisions and effective is not None and effective != (args.decisions == "on"):
        raise SystemExit(f"--decisions {args.decisions} not applied (settings say {effective})")

    t0 = time.perf_counter()
    runs = await get_outputs(cases, out_dir, args.refresh, use_langfuse, args.repeat)
    graph_s = time.perf_counter() - t0

    judge = QwenJudge()
    sem = asyncio.Semaphore(JUDGE_CONCURRENCY)
    t1 = time.perf_counter()

    async def judge_case(case: dict) -> dict:
        to_judge = runs[case["id"]] if args.judge_all else runs[case["id"]][:1]
        judged = await asyncio.gather(*(evaluate_case(case, o, judge, sem) for o in to_judge))
        return judged[0] if len(judged) == 1 else mean_metrics(list(judged))

    per_case = await asyncio.gather(*(judge_case(c) for c in cases))
    judge_s = time.perf_counter() - t1

    results: dict[str, dict] = {}
    for case, metrics in zip(cases, per_case, strict=True):
        main_out = runs[case["id"]][0]
        entry = {
            "score": (main_out.get("match") or {}).get("score"),
            "trace_id": main_out.get("trace_id"),
            "metrics": metrics,
            "run": run_stats(main_out),
        }
        if args.repeat:
            entry["stability"] = score_stability(runs[case["id"]])
            entry["repeat_runs"] = [run_stats(o) for o in runs[case["id"]]]
        results[case["id"]] = entry

    new_summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": {
            "label": args.label,
            "decisions": decisions,
            "repeat": args.repeat,
            "judge_all": args.judge_all,
            "judge_model": settings.llm_model,
        },
        "cases": [c["id"] for c in cases],
        "thresholds": THRESHOLDS,
        "graph_seconds": round(graph_s, 1),
        "judge_seconds": round(judge_s, 1),
        "results": results,
    }
    try:
        summary = merge_summaries(load_summary(ldir), new_summary, [c["id"] for c in all_cases])
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    ldir.mkdir(parents=True, exist_ok=True)
    (ldir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    disk_outputs, _ = load_label_outputs(ldir)
    write_report(ldir / "report.md", summary, {c["id"]: c for c in all_cases}, disk_outputs)

    if use_langfuse:
        main_outputs = {cid: r[0] for cid, r in runs.items()}
        logger.info("langfuse scores posted: %d", post_langfuse_scores(main_outputs, results))
    logger.info("averages (%d cases): %s", len(summary["cases"]), summary["averages"])
    logger.info("report: %s", ldir / "report.md")
    if summary["failed_thresholds"]:
        logger.error("thresholds failed: %s", summary["failed_thresholds"])
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cases", help="comma-separated case ids (default: all)")
    parser.add_argument(
        "--label",
        default=DEFAULT_LABEL,
        help="results go to evals/results/<label>/ (default: %(default)s)",
    )
    parser.add_argument(
        "--decisions",
        choices=["on", "off"],
        help="force DECISIONS_ENABLED for the graph run (default: settings / .env)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        metavar="N",
        help="run the graph N times per case (outputs/<case>__r<i>.json) for score stability",
    )
    parser.add_argument(
        "--judge-all", action="store_true", help="with --repeat: judge every run, not only r0"
    )
    parser.add_argument("--refresh", action="store_true", help="re-run the graph, ignore cache")
    parser.add_argument("--no-langfuse", action="store_true", help="no tracing, no scores")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.repeat is not None and args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")
    if args.judge_all and not args.repeat:
        raise SystemExit("--judge-all needs --repeat")
    apply_decisions_flag(args.decisions)  # before anything reads the settings
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "openai", "langfuse"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
