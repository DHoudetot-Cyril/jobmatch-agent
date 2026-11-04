"""evals/run_eval.py Langfuse score payloads (no network)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals import run_eval  # noqa: E402


def test_score_bodies_handle_summary_entries_with_ints_and_aggregates():
    outputs = {"a": {"trace_id": "t1"}, "b": {"trace_id": None}, "c": {"trace_id": "t3"}}
    results = {
        "a": {
            "score": 92,  # int: used to crash with "'int' object is not subscriptable"
            "trace_id": "t1",
            "metrics": {
                "skills_recall": {"value": 1.0, "reason": "all found"},
                "gaps_recall": {"value": None, "reason": "error"},
                "letter_faithfulness": {"value": 0.9, "reason": None, "values": [0.9, 0.8]},
                "broken": 3,
            },
            "run": {"elapsed_s": 100.0, "review_iterations": 2},
            "stability": {"std": 0.7, "scores": [80, 81]},
            "repeat_runs": [{"elapsed_s": 1.0}],
        },
        "b": {"score": 10, "metrics": {"skills_recall": {"value": 1.0}}},  # no trace -> skipped
        "c": {"skills_recall": {"value": 0.5, "reason": "legacy flat shape"}, "n": 4},
        "d": 7,  # unknown case / garbage
    }
    bodies = run_eval.langfuse_score_bodies(outputs, results)
    assert [(b["traceId"], b["name"], b["value"]) for b in bodies] == [
        ("t1", "eval_skills_recall", 1.0),
        ("t1", "eval_letter_faithfulness", 0.9),
        ("t3", "eval_skills_recall", 0.5),
    ]
    assert bodies[0]["id"] == "t1-skills_recall" and bodies[0]["dataType"] == "NUMERIC"
    assert bodies[1]["comment"] == ""


def test_post_langfuse_scores_unreachable_returns_zero(monkeypatch):
    from jobmatch.config import get_settings

    monkeypatch.setattr(get_settings(), "langfuse_host", "http://127.0.0.1:9")
    results = {"a": {"score": 1, "metrics": {"m": {"value": 1.0}}}}
    assert run_eval.post_langfuse_scores({"a": {"trace_id": "t"}}, results) == 0
