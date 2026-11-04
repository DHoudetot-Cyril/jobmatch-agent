"""Eval tooling (no LLM): summary merging, decision stats, score stability, A/B compare."""

import copy
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals import aggregate, compare, run_eval  # noqa: E402

BASELINE = ROOT / "evals" / "baseline"


def _metrics(v: float) -> dict:
    return {n: {"value": v, "reason": "x"} for n in aggregate.METRIC_ORDER}


def _summary(results: dict, decisions: str | None = "off") -> dict:
    return {
        "generated_at": "t",
        "config": {"decisions": decisions},
        "cases": list(results),
        "thresholds": {"letter_faithfulness": 0.7},
        "results": {c: {"score": 50, "metrics": _metrics(v)} for c, v in results.items()},
    }


def _output(score=50, skill_checks=None, review=None, trace=None, **kw) -> dict:
    match = {"score": score, "matched_skills": [], "gaps": []}
    if skill_checks is not None:
        match["skill_checks"] = skill_checks
    out = {"match": match, "review": review or {"approved": True}, "elapsed_s": 100.0, **kw}
    if trace is not None:
        out["trace"] = trace
    return out


# --------------------------------------------------------------------------- merging


def test_merge_partial_run_keeps_other_cases_and_recomputes_averages():
    full = _summary({"a": 1.0, "b": 1.0, "c": 1.0})
    partial = _summary({"b": 0.0})
    merged = aggregate.merge_summaries(full, partial, ["a", "b", "c"])
    assert merged["cases"] == ["a", "b", "c"]
    assert merged["results"]["b"]["metrics"]["skills_recall"]["value"] == 0.0
    assert merged["averages"]["skills_recall"] == pytest.approx(0.667, abs=1e-3)
    assert merged["failed_thresholds"] == ["letter_faithfulness"]  # 0.667 < 0.7
    assert len(merged["runs"]) == 2


def test_merge_without_existing_summary_and_threshold_failure():
    merged = aggregate.merge_summaries(None, _summary({"a": 0.5}), ["a"])
    assert merged["runs"][0]["cases"] == ["a"]
    assert merged["averages"]["skills_recall"] == 0.5
    assert merged["failed_thresholds"] == ["letter_faithfulness"]
    merged = aggregate.merge_summaries(merged, _summary({"b": 0.5}), ["b", "a"])
    assert merged["cases"] == ["b", "a"]  # dataset order
    assert aggregate.failed_thresholds(merged["averages"], {"letter_faithfulness": 0.7}) == [
        "letter_faithfulness"
    ]


def test_merge_refuses_mixed_decision_configs():
    with pytest.raises(ValueError, match="another --label"):
        aggregate.merge_summaries(_summary({"a": 1}, "off"), _summary({"a": 1}, "on"), ["a"])


def test_merge_into_baseline_style_summary():
    base = json.loads((BASELINE / "summary.json").read_text())
    merged = aggregate.merge_summaries(base, _summary({"strong_fit_backend": 0.0}), base["cases"])
    assert merged["cases"] == base["cases"]
    assert merged["averages"]["skills_recall"] == pytest.approx(7 / 8, abs=1e-3)


# --------------------------------------------------------------------------- stats


def test_baseline_output_has_no_decision_stats():
    out = json.loads((BASELINE / "outputs" / "strong_fit_backend.json").read_text())
    stats = aggregate.run_stats(out)
    assert stats["decisions"] is None
    assert stats["graph_seconds"] == out["elapsed_s"]
    assert stats["review_iterations"] == out["review_iterations"]
    assert stats["reviewer_approved"] is True


def test_decision_stats_new_fields():
    checks = [
        {"skill": "Python", "llm_credit": 1.0, "final_credit": 1.0, "agree": True},
        {"skill": "K8s", "llm_credit": 1.0, "final_credit": 0.0, "agree": False},
    ]
    review = {
        "approved": False,
        "unsupported_claims": ["led a team of 10"],
        "claim_checks": [
            {"claim": "led a team of 10", "p_supported": 0.1, "dismissed": False},
            {"claim": "knows SQL", "p_supported": 0.9, "dismissed": True},
        ],
    }
    out = _output(skill_checks=checks, review=review, trace=["score_match", "verify_skills"])
    d = aggregate.decision_stats(out)
    assert d["skill_checks"] == 2
    assert d["skill_agreement"] == 0.5
    assert d["skill_overrides"] == 1
    assert d["claims_flagged"] == 1
    assert d["claims_dismissed"] == 1
    assert d["decision_count"] == 4
    assert d["verify_nodes_in_trace"] == ["verify_skills"]
    # node in trace but no checks (decisions on, nothing to verify)
    assert aggregate.decision_stats(_output(trace=["verify_claims"]))["decision_count"] == 0


def test_score_stability():
    s = aggregate.score_stability([_output(40), _output(50), _output(60), {"error": "boom"}])
    assert (s["n"], s["mean"], s["std"], s["min"], s["max"]) == (3, 50, 10.0, 40, 60)
    assert s["failed_runs"] == 1
    assert aggregate.score_stability([_output(40)])["std"] is None
    assert aggregate.score_stability([{"error": "x"}]) is None


def test_load_label_outputs_groups_repeats(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    for i, score in enumerate([70, 60, 80]):
        (out / f"case_a__r{i}.json").write_text(json.dumps(_output(score)))
    (out / "case_b.json").write_text(json.dumps(_output(10)))
    main, repeats = aggregate.load_label_outputs(tmp_path)
    assert main["case_a"]["match"]["score"] == 70  # r0
    assert [o["match"]["score"] for o in repeats["case_a"]] == [70, 60, 80]
    assert main["case_b"]["match"]["score"] == 10 and "case_b" not in repeats


# --------------------------------------------------------------------------- compare


def test_compare_baseline_with_itself_has_zero_deltas():
    side = compare.load_side(BASELINE)
    report = compare.build_report(side, side, compare.expected_ranges())
    assert "Cases compared: 8" in report
    assert "better |" not in report and "worse |" not in report
    assert "letter_faithfulness` | 0.925 | 0.925 | 0.000 | = |" in report
    assert "no decision-layer data (n/a)" in report


def test_compare_detects_changes(tmp_path):
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    shutil.copytree(BASELINE, a_dir)
    shutil.copytree(BASELINE, b_dir)
    summary = json.loads((b_dir / "summary.json").read_text())
    summary["results"]["strong_fit_backend"]["metrics"]["letter_faithfulness"]["value"] = 0.1
    (b_dir / "summary.json").write_text(json.dumps(summary))
    path = b_dir / "outputs" / "strong_fit_backend.json"
    out = json.loads(path.read_text())
    out["match"]["score"] = 50  # out of the 75-100 range
    out["match"]["skill_checks"] = [{"skill": "x", "llm_credit": 1, "final_credit": 1, "agree": 1}]
    out["review"]["claim_checks"] = [{"claim": "c", "p_supported": 0.2, "dismissed": False}]
    out["review"]["unsupported_claims"] = ["c"]
    path.write_text(json.dumps(out))
    for i, score in enumerate([50, 54]):
        rep = copy.deepcopy(out)
        rep["match"]["score"] = score
        (b_dir / "outputs" / f"strong_fit_backend__r{i}.json").write_text(json.dumps(rep))

    report = compare.build_report(
        compare.load_side(a_dir), compare.load_side(b_dir), compare.expected_ranges()
    )
    assert "| `letter_faithfulness` | 0.925 | 0.825 | -0.100 | ↓ worse |" in report
    assert "| strong_fit_backend | 75-100 | 94 | 50 (out of range) | -44 |" in report
    assert "52.0 ± 2.83 [50-54] (2)" in report
    assert "2 jevos decisions on 1 cases" in report
    assert "1 flagged unsupported" in report


def test_compare_cli_writes_file(tmp_path, capsys):
    out = tmp_path / "cmp.md"
    assert compare.main([str(BASELINE), str(BASELINE), "--out", str(out)]) == 0
    assert out.read_text().startswith("# Eval comparison: baseline vs baseline")


# --------------------------------------------------------------------------- run_eval


def test_run_eval_report_renders_baseline_summary(tmp_path):
    summary = json.loads((BASELINE / "summary.json").read_text())
    outputs, _ = aggregate.load_label_outputs(BASELINE)
    cases = {c["id"]: c for c in run_eval.load_cases()}
    run_eval.write_report(tmp_path / "report.md", summary, cases, outputs)
    text = (tmp_path / "report.md").read_text()
    assert "| strong_fit_backend | 94 (75-100) |" in text
    assert "**0.93**" in text


def test_run_eval_cli_options_and_paths(tmp_path):
    args = run_eval.build_parser().parse_args(
        ["--label", "jevos", "--decisions", "on", "--repeat", "3", "--judge-all"]
    )
    assert (args.label, args.decisions, args.repeat, args.judge_all) == ("jevos", "on", 3, True)
    assert run_eval.build_parser().parse_args([]).label == "latest"
    assert run_eval.output_path(tmp_path, "c", None).name == "c.json"
    assert run_eval.output_path(tmp_path, "c", 2).name == "c__r2.json"


def test_apply_decisions_flag_sets_env(monkeypatch):
    from jobmatch.config import get_settings

    # setenv first so monkeypatch restores the original value after the test
    monkeypatch.setenv("DECISIONS_ENABLED", "unset")
    monkeypatch.setattr(get_settings, "cache_clear", lambda: None)
    run_eval.apply_decisions_flag("off")
    assert os.environ["DECISIONS_ENABLED"] == "false"
    run_eval.apply_decisions_flag("on")
    assert os.environ["DECISIONS_ENABLED"] == "true"
