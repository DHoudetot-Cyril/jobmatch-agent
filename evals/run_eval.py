"""Evaluation runner: graph run (cached) -> metrics -> report + Langfuse scores.

    uv run --group eval python -m evals.run_eval [--cases id1,id2] [--refresh] [--no-langfuse]

1. For each dataset case, run the real graph (MemorySaver) up to the human_review interrupt,
   at most `GRAPH_CONCURRENCY` cases at a time (llama-server has 2 slots). Outputs are cached
   in `evals/results/outputs/<case>.json` and reused unless `--refresh`.
2. Compute deterministic metrics + G-Eval metrics (local Qwen judge).
3. Write `evals/results/report.md` and `evals/results/summary.json`, post every metric as a
   Langfuse score on the case's trace (skipped silently if Langfuse is down / --no-langfuse).
4. Exit 1 if an average falls below `THRESHOLDS`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

from evals.judge import QwenJudge
from evals.metrics import build_geval_metrics, deterministic_metrics, geval_test_cases
from jobmatch.config import get_settings
from jobmatch.schemas import MatchResult

# Averages below these values make the run fail (exit code 1).
THRESHOLDS: dict[str, float] = {
    "letter_faithfulness": 0.7,
    "score_in_range": 0.6,
}
GRAPH_CONCURRENCY = 2
JUDGE_CONCURRENCY = 2

EVALS_DIR = Path(__file__).parent
DATASET_DIR = EVALS_DIR / "dataset"
RESULTS_DIR = EVALS_DIR / "results"
OUTPUTS_DIR = RESULTS_DIR / "outputs"

METRIC_ORDER = [
    "score_in_range",
    "skills_recall",
    "gaps_recall",
    "letter_faithfulness",
    "letter_relevance",
    "gap_usefulness",
]

logger = logging.getLogger("evals")


# --------------------------------------------------------------------------- dataset


def load_cases(selected: list[str] | None) -> list[dict]:
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


async def run_case(case: dict, use_langfuse: bool) -> dict:
    """Run the graph up to the human_review interrupt and return a JSON-able output."""
    from langgraph.checkpoint.memory import MemorySaver

    from jobmatch.graph.build import build_graph
    from jobmatch.observability.tracing import get_langfuse_handler, langfuse_run_config

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    # One trace per (case, graph run); the seed is cached so scores land on the right trace.
    trace_seed = f"eval-{case['id']}-{stamp}"
    analysis_id = f"eval-{case['id']}-{uuid.uuid4().hex[:8]}"
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
        "offer": _dump(values.get("offer")),
        "profile": _dump(values.get("profile")),
        "match": _dump(values.get("match")),
        "cover_letter": _dump(values.get("cover_letter")),
        "review": _dump(values.get("review")),
        "review_iterations": values.get("review_iterations"),
        "status": str(values.get("status")),
    }


async def get_outputs(cases: list[dict], refresh: bool, use_langfuse: bool) -> dict[str, dict]:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(GRAPH_CONCURRENCY)
    outputs: dict[str, dict] = {}

    async def one(case: dict) -> None:
        path = OUTPUTS_DIR / f"{case['id']}.json"
        if path.exists() and not refresh:
            outputs[case["id"]] = json.loads(path.read_text())
            logger.info("cache hit: %s", case["id"])
            return
        async with sem:
            logger.info("graph run: %s ...", case["id"])
            try:
                out = await run_case(case, use_langfuse)
            except Exception as exc:  # one failing case must not kill the whole run
                logger.exception("graph run failed: %s", case["id"])
                outputs[case["id"]] = {"case_id": case["id"], "error": repr(exc)}
                return
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
        outputs[case["id"]] = out
        logger.info("graph done: %s in %.0fs", case["id"], out["elapsed_s"])

    await asyncio.gather(*(one(c) for c in cases))
    if use_langfuse:
        from jobmatch.observability.tracing import flush_langfuse

        flush_langfuse()
    return outputs


# --------------------------------------------------------------------------- judging


async def evaluate_case(case: dict, output: dict, judge: QwenJudge, sem: asyncio.Semaphore):
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


# --------------------------------------------------------------------------- Langfuse


def post_langfuse_scores(outputs: dict[str, dict], results: dict[str, dict]) -> int:
    """POST /api/public/scores for every metric. Returns the number of scores posted."""
    s = get_settings()
    auth = (s.langfuse_public_key, s.langfuse_secret_key)
    url = f"{s.langfuse_host.rstrip('/')}/api/public/scores"
    posted = 0
    try:
        with httpx.Client(timeout=5.0, auth=auth) as client:
            for case_id, metrics in results.items():
                trace_id = outputs[case_id].get("trace_id")
                if not trace_id:
                    continue
                for name, m in metrics.items():
                    if m["value"] is None:
                        continue
                    body = {
                        # deterministic id: re-judging overwrites instead of duplicating
                        "id": f"{trace_id}-{name}",
                        "traceId": trace_id,
                        "name": f"eval_{name}",
                        "value": m["value"],
                        "dataType": "NUMERIC",
                        "comment": (m.get("reason") or "")[:1000],
                    }
                    resp = client.post(url, json=body)
                    if resp.status_code >= 300:
                        logger.warning("score rejected %s: %s", resp.status_code, resp.text)
                        continue
                    posted += 1
    except httpx.HTTPError as exc:
        logger.info("langfuse unreachable, scores skipped (%s)", exc)
    return posted


# --------------------------------------------------------------------------- report


def averages(results: dict[str, dict]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for name in METRIC_ORDER:
        vals = [r[name]["value"] for r in results.values() if r[name]["value"] is not None]
        out[name] = round(sum(vals) / len(vals), 3) if vals else None
    return out


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.2f}"


def write_report(cases, outputs, results, avgs, failures, judge_s) -> None:
    lines = [
        "# JobMatch evaluation report",
        "",
        f"Generated {datetime.now(UTC).isoformat(timespec='seconds')} · judge: "
        f"{get_settings().llm_model} (local) · cases: {len(cases)} · judging {judge_s:.0f}s",
        "",
        "| case | score (expected) | in range | skills recall | gaps recall | faithfulness "
        "| relevance | gap usefulness |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for case in cases:
        out, r = outputs[case["id"]], results[case["id"]]
        score = out.get("match", {}).get("score", "—") if "error" not in out else "ERROR"
        lo, hi = case["expected_score_range"]
        cells = " | ".join(_fmt(r[n]["value"]) for n in METRIC_ORDER)
        lines.append(f"| {case['id']} | {score} ({lo}-{hi}) | {cells} |")
    avg_cells = " | ".join(f"**{_fmt(avgs[n])}**" for n in METRIC_ORDER)
    lines.append(f"| **average** | | {avg_cells} |")
    lines += ["", "## Thresholds", ""]
    for name, thr in THRESHOLDS.items():
        status = "FAIL" if name in failures else "ok"
        lines.append(f"- `{name}` average {_fmt(avgs[name])} (>= {thr}): **{status}**")
    lines += ["", "## Details", ""]
    for case in cases:
        out, r = outputs[case["id"]], results[case["id"]]
        lines += [f"### {case['id']}", "", f"_{case.get('description', '')}_", ""]
        if "error" in out:
            lines += [f"Graph error: `{out['error']}`", ""]
            continue
        match = out["match"]
        lines += [
            f"- graph run: {out.get('elapsed_s')}s, review iterations "
            f"{out.get('review_iterations')}, reviewer approved: "
            f"{(out.get('review') or {}).get('approved')}",
            f"- matched skills: {', '.join(match['matched_skills']) or '(none)'}",
            f"- gaps: {', '.join(g['skill'] for g in match['gaps']) or '(none)'}",
            f"- letter language: {out['cover_letter'].get('language')}",
        ]
        for name in METRIC_ORDER:
            reason = (r[name].get("reason") or "").replace("\n", " ")
            lines.append(f"- **{name}** {_fmt(r[name]['value'])} — {reason}")
        lines.append("")
    (RESULTS_DIR / "report.md").write_text("\n".join(lines))


# --------------------------------------------------------------------------- main


async def main_async(args: argparse.Namespace) -> int:
    selected = [c.strip() for c in args.cases.split(",")] if args.cases else None
    cases = load_cases(selected)
    use_langfuse = not args.no_langfuse and get_settings().langfuse_enabled
    t0 = time.perf_counter()
    outputs = await get_outputs(cases, args.refresh, use_langfuse)
    graph_s = time.perf_counter() - t0

    judge = QwenJudge()
    sem = asyncio.Semaphore(JUDGE_CONCURRENCY)
    t1 = time.perf_counter()
    per_case = await asyncio.gather(
        *(evaluate_case(c, outputs[c["id"]], judge, sem) for c in cases)
    )
    judge_s = time.perf_counter() - t1
    results = {c["id"]: r for c, r in zip(cases, per_case, strict=True)}
    avgs = averages(results)
    failures = [n for n, thr in THRESHOLDS.items() if avgs[n] is None or avgs[n] < thr]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_report(cases, outputs, results, avgs, failures, judge_s)
    summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "cases": [c["id"] for c in cases],
        "averages": avgs,
        "thresholds": THRESHOLDS,
        "failed_thresholds": failures,
        "graph_seconds": round(graph_s, 1),
        "judge_seconds": round(judge_s, 1),
        "results": {
            cid: {
                "score": outputs[cid].get("match", {}).get("score"),
                "trace_id": outputs[cid].get("trace_id"),
                "metrics": r,
            }
            for cid, r in results.items()
        },
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    if use_langfuse:
        logger.info("langfuse scores posted: %d", post_langfuse_scores(outputs, results))
    logger.info("averages: %s", avgs)
    logger.info("report: %s", RESULTS_DIR / "report.md")
    if failures:
        logger.error("thresholds failed: %s", failures)
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cases", help="comma-separated case ids (default: all)")
    parser.add_argument("--refresh", action="store_true", help="re-run the graph, ignore cache")
    parser.add_argument("--no-langfuse", action="store_true", help="no tracing, no scores")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "openai", "langfuse"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
