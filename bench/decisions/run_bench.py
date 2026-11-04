"""Benchmark "System 1" decision backends on the labelled decision dataset.

    uv run python bench/decisions/run_bench.py --backends qwen-logits,qwen-logits-perm \
        --tasks skill_evidence,claim_supported --concurrency 2
    uv run --group decisions python bench/decisions/run_bench.py --backends laya,jevos

Each backend runs in its own subprocess (so peak RSS / load time are per backend), unless
--in-process. Raw probabilities are cached in results/raw/<backend>.jsonl and reused on the
next run (resume-able; items that errored are retried). Then metrics are computed per
backend x task x lang on the TEST split, raw and after temperature scaling fitted on the
CALIB split, and written to results/report.md + results/summary.json.
A run name `<backend>@q-en` asks the English version of every question (state unchanged): an
ablation separating "document language" from "question language".
Use --report-only to recompute the report from cached raw results.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import resource
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import metrics as M  # noqa: E402

DATASET = HERE / "dataset"
RESULTS = HERE / "results"
RAW = RESULTS / "raw"
TASKS = ["skill_evidence", "claim_supported", "seniority", "doc_type"]
BUDGET = 0.05

log = logging.getLogger("bench.decisions")


# ------------------------------------------------------------------------------------ data
def load_items(tasks: list[str], limit: int | None = None) -> list[dict]:
    items = []
    for t in tasks:
        rows = [json.loads(line) for line in (DATASET / f"{t}.jsonl").read_text().splitlines()]
        items.extend(rows[:limit] if limit else rows)
    return items


def read_raw(backend: str) -> dict[str, dict]:
    path = RAW / f"{backend}.jsonl"
    out: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r  # last write wins
    return out


def _children_pids() -> list[int]:
    pids: list[int] = []
    try:
        for task in Path("/proc/self/task").iterdir():
            pids += [int(x) for x in (task / "children").read_text().split()]
    except OSError:
        pass
    return pids


def _children_rss_mb(field: str = "VmHWM") -> float | None:
    """Sum of `field` over live child processes (e.g. a spawned `jev serve`)."""
    vals = [_rss_mb(field, pid) for pid in _children_pids()]
    vals = [v for v in vals if v is not None]
    return sum(vals) if vals else None


def _rss_mb(field: str = "VmRSS", pid: int | str = "self") -> float | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(field + ":"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


# ------------------------------------------------------------------------------------ worker
async def run_backend(backend: str, items: list[dict], concurrency: int) -> dict:
    from jobmatch.decisions.base import Question
    from jobmatch.decisions.registry import get_backend

    RAW.mkdir(parents=True, exist_ok=True)
    done = {k for k, r in read_raw(backend).items() if not r.get("error")}
    todo = [it for it in items if it["id"] not in done]
    # group by state so llama.cpp prompt caching (and any backend-side cache) can help
    todo.sort(key=lambda it: (it["state"], it["id"]))
    meta: dict = {"backend": backend, "rss_before_mb": _rss_mb()}
    log.info("%s: %d items cached, %d to run", backend, len(items) - len(todo), len(todo))

    t0 = time.perf_counter()
    base, _, variant = backend.partition("@")
    qfield = "question_en" if variant == "q-en" else "question"
    model = get_backend(base)
    if todo:  # warm-up call so lazy model loading is counted as load time, not latency
        it = todo[0]
        try:
            await model.decide(it["state"], Question(**it[qfield]))
        except ImportError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("%s warm-up failed: %s", backend, exc)
    meta["load_s"] = time.perf_counter() - t0
    meta["rss_after_load_mb"] = _rss_mb()
    meta["children_rss_after_load_mb"] = _children_rss_mb("VmRSS")

    sem = asyncio.Semaphore(concurrency)
    path = RAW / f"{backend}.jsonl"
    n_done = 0

    async def one(it: dict, fh) -> None:
        nonlocal n_done
        async with sem:
            rec = {"id": it["id"], "task": it["task"]}
            try:
                d = await model.decide(it["state"], Question(**it[qfield]))
                rec.update(probs=d.probs, latency_ms=d.latency_ms)
            except Exception as exc:  # noqa: BLE001
                rec.update(error=f"{type(exc).__name__}: {exc}"[:500])
                log.warning("%s %s failed: %s", backend, it["id"], rec["error"])
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n_done += 1
            if n_done % 10 == 0:
                log.info("%s: %d/%d", backend, n_done, len(todo))

    t1 = time.perf_counter()
    with path.open("a") as fh:
        await asyncio.gather(*(one(it, fh) for it in todo))
    meta["run_s"] = time.perf_counter() - t1
    live_children_hwm = _children_rss_mb("VmHWM")  # before aclose() stops a spawned server
    await model.aclose()
    meta["peak_rss_mb"] = _rss_mb("VmHWM")
    reaped = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    meta["children_peak_rss_mb"] = max(live_children_hwm or 0.0, reaped) or None
    meta["n_run"] = len(todo)
    meta["concurrency"] = concurrency
    return meta


def worker_main(args: argparse.Namespace) -> int:
    items = load_items(args.tasks, args.limit)
    try:
        meta = asyncio.run(run_backend(args.worker, items, args.concurrency))
    except ImportError as exc:
        log.error("backend %s not importable: %s", args.worker, exc)
        return 3
    if meta.get("n_run"):  # keep the meta of the run that actually did the work
        (RAW / f"{args.worker}.meta.json").write_text(json.dumps(meta, indent=2))
    return 0


# ------------------------------------------------------------------------------------ report
def _labels_of(item: dict) -> list[str]:
    q = item["question"]
    return ["no", "yes"] if q["kind"] == "yes_no" else list(q["options"])


def evaluate(backend: str, items: list[dict], raw: dict[str, dict]) -> dict:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_task[it["task"]].append(it)
    out: dict = {}
    for task, rows in by_task.items():
        labels = _labels_of(rows[0])
        got = [(it, raw[it["id"]]) for it in rows if it["id"] in raw and "probs" in raw[it["id"]]]
        n_err = sum(1 for it in rows if it["id"] in raw and raw[it["id"]].get("error"))
        calib = [(it, r) for it, r in got if it["split"] == "calib"]
        t = M.fit_temperature([r["probs"] for _, r in calib], [it["label"] for it, _ in calib])
        res_task: dict = {"temperature": t, "n_missing": len(rows) - len(got), "n_error": n_err}
        for lang in ["all", "fr", "en"]:
            test = [
                (it, r)
                for it, r in got
                if it["split"] == "test" and (lang == "all" or it["lang"] == lang)
            ]
            if not test:
                continue
            gold = [it["label"] for it, _ in test]
            raw_p = [{lab: r["probs"].get(lab, 0.0) for lab in labels} for _, r in test]
            cal_p = [M.apply_temperature(p, t) for p in raw_p]
            lat = [r.get("latency_ms", 0.0) for _, r in test if r.get("latency_ms")]
            res_task[lang] = {
                "n": len(test),
                "accuracy": M.accuracy(raw_p, gold),
                "macro_f1": M.macro_f1(raw_p, gold, labels),
                "brier_raw": M.brier(raw_p, gold),
                "ece_raw": M.ece(raw_p, gold),
                "brier_cal": M.brier(cal_p, gold),
                "ece_cal": M.ece(cal_p, gold),
                "auto5_raw": M.automatable_share(raw_p, gold, BUDGET),
                "auto5_cal": M.automatable_share(cal_p, gold, BUDGET),
                "latency_p50_ms": M.percentile(lat, 0.5) if lat else None,
                "latency_p95_ms": M.percentile(lat, 0.95) if lat else None,
            }
        out[task] = res_task
    return out


def majority_baseline(items: list[dict]) -> dict:
    """Predict the calib-split class frequencies for every test item."""
    raw = {}
    by_task: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_task[it["task"]].append(it)
    for rows in by_task.values():
        labels = _labels_of(rows[0])
        cnt = Counter(it["label"] for it in rows if it["split"] == "calib")
        tot = sum(cnt.values()) + len(labels) * 0.5  # add-1/2 smoothing
        probs = {lab: (cnt[lab] + 0.5) / tot for lab in labels}
        for it in rows:
            raw[it["id"]] = {"probs": probs, "latency_ms": 0.0}
    return raw


def _f(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "–"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def write_report(summary: dict, items: list[dict]) -> None:
    lines = [
        "# Decision models benchmark",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Metrics on the **test** split; "
        "'cal' = after temperature scaling fitted per backend × task on the **calib** split. "
        f"auto@5% = largest share of test items that can be answered automatically (most "
        f"confident first) with ≤ {BUDGET:.0%} errors. See README.md for definitions.",
        "",
    ]
    counts = Counter((it["task"], it["lang"], it["split"]) for it in items)
    lines += ["## Dataset", "", "| task | lang | calib | test |", "|---|---|---|---|"]
    for task in sorted({it["task"] for it in items}):
        for lang in ["fr", "en"]:
            lines.append(
                f"| {task} | {lang} | {counts[(task, lang, 'calib')]} | "
                f"{counts[(task, lang, 'test')]} |"
            )
    lines.append("")

    hdr = (
        "| backend | lang | n | acc | macro-F1 | Brier raw | ECE raw | Brier cal | ECE cal | T "
        "| auto@5% raw | auto@5% cal | p50 ms | p95 ms |"
    )
    sep = "|" + "---|" * 14
    for task in [t for t in TASKS if any(t in r["metrics"] for r in summary.values())]:
        lines += [f"## {task}", "", hdr, sep]
        for be, res in summary.items():
            m = res["metrics"].get(task)
            if not m:
                continue
            for lang in ["all", "fr", "en"]:
                r = m.get(lang)
                if not r:
                    continue
                lines.append(
                    f"| {be} | {lang} | {r['n']} | {_f(r['accuracy'])} | {_f(r['macro_f1'])} | "
                    f"{_f(r['brier_raw'])} | {_f(r['ece_raw'])} | {_f(r['brier_cal'])} | "
                    f"{_f(r['ece_cal'])} | {_f(m['temperature'], 2)} | {_f(r['auto5_raw'], 2)} | "
                    f"{_f(r['auto5_cal'], 2)} | {_f(r['latency_p50_ms'], 0)} | "
                    f"{_f(r['latency_p95_ms'], 0)} |"
                )
        lines.append("")

    lines += [
        "## Resources",
        "",
        "| backend | load s | run s (cached run) | conc. | RSS after load MB | peak RSS MB | "
        "children peak RSS MB | errors | missing |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for be, res in summary.items():
        meta = res.get("meta") or {}
        n_err = sum(m.get("n_error", 0) for m in res["metrics"].values())
        n_miss = sum(m.get("n_missing", 0) for m in res["metrics"].values())
        lines.append(
            f"| {be} | {_f(meta.get('load_s'), 1)} | {_f(meta.get('run_s'), 0)} | "
            f"{meta.get('concurrency', '–')} | {_f(meta.get('rss_after_load_mb'), 0)} | "
            f"{_f(meta.get('peak_rss_mb'), 0)} | {_f(meta.get('children_peak_rss_mb'), 0)} | "
            f"{n_err} | {n_miss} |"
        )
    lines += [
        "",
        "RSS = the benchmark worker process (Python + client). Children = processes the backend "
        "spawned (e.g. `jev serve`): RSS after load and peak (VmHWM). A server that was already "
        "running before the benchmark is not measured. Qwen (if run) lives in llama-server on "
        "the GPU: not measured.",
        "",
    ]
    lines += verdict(summary)
    lines += worst_errors(summary, items)
    lines += ["## Observations (auto-generated)", ""]
    lines += observations(summary)
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n")


YES_NO_TASKS = ["skill_evidence", "claim_supported"]
BAR_ACC, BAR_AUTO = 0.85, 0.5


def verdict(summary: dict) -> list[str]:
    """Per backend: is it 'very good' in BOTH French and English on the yes/no tasks?"""
    out = [
        "## Verdict",
        "",
        f"Bar for 'very good': on each yes/no task ({', '.join(YES_NO_TASKS)}) and EACH language, "
        f"test accuracy ≥ {BAR_ACC} and automatable share at 5% error (after calibration) ≥ "
        f"{BAR_AUTO}. Choice tasks are reported for information.",
        "",
        "| backend | task | lang | n | acc | acc 95% CI low (Wilson) | auto@5% cal | pass |",
        "|---|---|---|---|---|---|---|---|",
    ]
    concl = []
    for be, res in summary.items():
        if be == "majority-baseline":
            continue
        fails = []
        for task in YES_NO_TASKS + ["seniority", "doc_type"]:
            for lang in ["fr", "en"]:
                r = res["metrics"].get(task, {}).get(lang)
                if not r:
                    if task in YES_NO_TASKS:
                        fails.append(f"{task}/{lang} (no results)")
                    continue
                ok = r["accuracy"] >= BAR_ACC and r["auto5_cal"] >= BAR_AUTO
                mark = (
                    ("✅" if ok else "❌") if task in YES_NO_TASKS else ("(✅)" if ok else "(❌)")
                )
                if task in YES_NO_TASKS and not ok:
                    fails.append(
                        f"{task}/{lang} (acc {r['accuracy']:.2f}, auto {r['auto5_cal']:.2f})"
                    )
                out.append(
                    f"| {be} | {task} | {lang} | {r['n']} | {r['accuracy']:.3f} | "
                    f"{M.wilson_low(r['accuracy'], r['n']):.2f} | {r['auto5_cal']:.2f} | {mark} |"
                )
        if fails:
            concl.append(
                f"- **{be}: NOT 'very good' in both FR and EN.** Below the bar: "
                + "; ".join(fails)
                + "."
            )
        else:
            concl.append(f"- **{be}: meets the bar in BOTH French and English** on yes/no tasks.")
    passed = {
        be
        for be, line in zip([b for b in summary if b != "majority-baseline"], concl, strict=True)
        if "meets the bar" in line
    }
    for be in passed:
        base = be.partition("@")[0]
        if be != base and base in summary and base not in passed:
            concl.append(
                f"- `{base}` fails but `{be}` passes: the weakness is the QUESTION language, not "
                "the document language — French documents work when the question and option "
                "descriptions are written in English (which the graph controls)."
            )
    concl.append(
        "- English test sets are tiny (n ≤ 13 per task): a perfect score on 7 items still has a "
        "95% lower bound around 0.65 — see the Wilson column before trusting 'EN passes'."
    )
    return out + [""] + concl + [""]


def worst_errors(summary: dict, items: list[dict], k: int = 10) -> list[str]:
    """The k most confident wrong answers per backend (all splits) — label spot-check list."""
    by_id = {it["id"]: it for it in items}
    out = []
    for be in summary:
        if be == "majority-baseline":
            continue
        raw = read_raw(be)
        errs = []
        for i, r in raw.items():
            it = by_id.get(i)
            if not it or "probs" not in r:
                continue
            pred = M.argmax(r["probs"])
            if pred != it["label"]:
                errs.append((r["probs"][pred], it, pred))
        errs.sort(key=lambda e: -e[0])
        out += [
            f"## {be}: {k} most confident errors (all splits, for label spot-check)",
            "",
            "| id | task | lang | split | label | predicted | confidence | label note |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for conf, it, pred in errs[:k]:
            note = (it.get("note") or "").replace("|", "/")
            out.append(
                f"| {it['id']} | {it['task']} | {it['lang']} | {it['split']} | {it['label']} | "
                f"{pred} | {conf:.3f} | {note} |"
            )
        out.append("")
    return out


def observations(summary: dict) -> list[str]:
    obs = []
    for task in TASKS:
        rows = [
            (be, res["metrics"][task]["all"])
            for be, res in summary.items()
            if task in res["metrics"] and "all" in res["metrics"][task]
        ]
        if not rows:
            continue
        best = max(rows, key=lambda r: r[1]["macro_f1"])
        best_auto = max(rows, key=lambda r: r[1]["auto5_cal"])
        obs.append(
            f"- **{task}**: best macro-F1 {best[0]} ({best[1]['macro_f1']:.3f}, acc "
            f"{best[1]['accuracy']:.3f}); highest automatable share at 5% error "
            f"{best_auto[0]} ({best_auto[1]['auto5_cal']:.0%} of test items)."
        )
        for be, r in rows:
            if r["ece_raw"] - r["ece_cal"] > 0.05:
                obs.append(
                    f"  - {be}: temperature scaling cuts ECE {r['ece_raw']:.3f} → "
                    f"{r['ece_cal']:.3f} (T={summary[be]['metrics'][task]['temperature']:.2f})."
                )
            elif r["ece_cal"] - r["ece_raw"] > 0.03:
                obs.append(
                    f"  - {be}: calibration fitted on the small calib split HURTS on test "
                    f"(ECE {r['ece_raw']:.3f} → {r['ece_cal']:.3f})."
                )
        for be, res in summary.items():
            m = res["metrics"].get(task, {})
            if "fr" in m and "en" in m and abs(m["fr"]["accuracy"] - m["en"]["accuracy"]) > 0.1:
                obs.append(
                    f"  - {be}: FR/EN accuracy gap {m['fr']['accuracy']:.2f} vs "
                    f"{m['en']['accuracy']:.2f} (EN n={m['en']['n']}, small)."
                )
    if any(be.startswith("qwen-logits") for be in summary) and {
        "qwen-logits",
        "qwen-logits-perm",
    } <= set(summary):
        for task in TASKS:
            a = summary["qwen-logits"]["metrics"].get(task, {}).get("all")
            b = summary["qwen-logits-perm"]["metrics"].get(task, {}).get("all")
            if a and b:
                obs.append(
                    f"- permutation averaging on {task}: acc {a['accuracy']:.3f} → "
                    f"{b['accuracy']:.3f}, ECE raw {a['ece_raw']:.3f} → {b['ece_raw']:.3f}, "
                    f"p50 {_f(a['latency_p50_ms'], 0)} → {_f(b['latency_p50_ms'], 0)} ms."
                )
    obs.append(
        "- Test splits are small (EN especially): differences of a few points are within noise."
    )
    return obs


def build_summary(backends: list[str], items: list[dict]) -> dict:
    summary = {
        "majority-baseline": {"metrics": evaluate("majority", items, majority_baseline(items))}
    }
    for be in backends:
        raw = read_raw(be)
        if not raw:
            continue
        meta_p = RAW / f"{be}.meta.json"
        summary[be] = {
            "metrics": evaluate(be, items, raw),
            "meta": json.loads(meta_p.read_text()) if meta_p.exists() else None,
        }
    return summary


# ------------------------------------------------------------------------------------ main
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--backends", default="jevos")
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--limit", type=int, default=None, help="max items per task")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--in-process", action="store_true", help="do not spawn one process/backend")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--worker", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    a.tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    a.backends = [b.strip() for b in a.backends.split(",") if b.strip()]
    return a


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = parse_args(argv)
    if args.worker:
        return worker_main(args)

    from jobmatch.decisions.registry import canonical_name

    backends = []
    for b in args.backends:  # "<backend>[@q-en]": @q-en = ask every question in English
        base, sep, variant = b.partition("@")
        if variant not in ("", "q-en"):
            raise SystemExit(f"unknown variant @{variant} (only @q-en)")
        backends.append(canonical_name(base) + sep + variant)
    items = load_items(args.tasks, args.limit)
    if not args.report_only:
        for be in backends:
            if args.in_process:
                try:
                    meta = asyncio.run(run_backend(be, items, args.concurrency))
                    if meta.get("n_run"):
                        (RAW / f"{be}.meta.json").write_text(json.dumps(meta, indent=2))
                except ImportError as exc:
                    log.error("skipping backend %s (not importable): %s", be, exc)
                continue
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                be,
                "--tasks",
                ",".join(args.tasks),
                "--concurrency",
                str(args.concurrency),
            ]
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            rc = subprocess.call(cmd, env=os.environ.copy())
            if rc == 3:
                log.error("skipping backend %s: not importable (see log above)", be)
            elif rc:
                log.error("backend %s worker exited with code %d", be, rc)

    summary = build_summary(backends, items)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    write_report(summary, items)
    log.info("wrote %s and %s", RESULTS / "report.md", RESULTS / "summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
