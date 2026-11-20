"""Extract a replayable record of one real analysis, for the demo GIF.

Reads the analysis from the API and its trace from Langfuse (v4 `v2/observations` endpoint),
then keeps only what the demo shows: graph nodes and tool calls with their real start offset,
duration and a one-line summary of their output. No prompts, no raw LLM messages.

    uv run python scripts/demo_extract.py <analysis_id> -o docs/demo/run.json

Defaults match the local stack (.env.example). Override with JOBMATCH_API_URL, LANGFUSE_HOST,
LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

NODES = {
    "supervisor", "parse_offer", "parse_cv", "score_match", "verify_skills", "analyze_gaps",
    "career_advisor", "career_tools", "career_plan", "write_letter", "review_letter",
    "verify_claims", "human_review",
}  # fmt: skip
TOOLS = {"search_skill_taxonomy", "check_cv_evidence", "search_learning_resources"}


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def fetch_observations(analysis_id: str, host: str, auth: tuple[str, str]) -> list[dict]:
    """All observations of the analysis trace(s), found through their Langfuse session id."""
    url = f"{host}/api/public/v2/observations"
    params = {"limit": 1000, "fields": "core,basic,io"}
    recent = httpx.get(url, params=params, auth=auth, timeout=30).raise_for_status().json()["data"]
    trace_ids = {o["traceId"] for o in recent if o.get("sessionId") == analysis_id}
    if not trace_ids:
        raise SystemExit(f"No Langfuse trace found for analysis {analysis_id}.")
    observations = []
    for trace_id in trace_ids:
        resp = httpx.get(url, params={**params, "traceId": trace_id}, auth=auth, timeout=30)
        observations += resp.raise_for_status().json()["data"]
    return sorted(observations, key=lambda o: o["startTime"])


# ---------------------------------------------------------------------- one-line summaries
def node_detail(name: str, out: Any) -> str:
    out = out if isinstance(out, dict) else {}
    if name == "parse_offer" and "offer" in out:
        o = out["offer"]
        required = sum(s["level"] == "required" for s in o["skills"])
        return f"{o['title']} · {o['company']} · {len(o['skills'])} skills ({required} required)"
    if name == "parse_cv" and "profile" in out:
        p = out["profile"]
        return f"{p['name']} · {len(p['skills'])} skills · {len(p['experiences'])} experiences"
    if name == "score_match" and "match" in out:
        m = out["match"]
        return (
            f"score {m['score']}/100 · {len(m['matched_skills'])} matched · {len(m['gaps'])} gaps"
        )
    if name == "verify_skills" and "match" in out:
        checks = out["match"].get("skill_checks") or []
        agree = sum(c.get("agree") for c in checks)
        return f"jevos checked {len(checks)} skills · agrees with the LLM on {agree}"
    if name == "analyze_gaps" and "match" in out:
        gaps = out["match"]["gaps"]
        return "gaps: " + ", ".join(f"{g['skill']} ({g['level'].replace('_', '-')})" for g in gaps)
    if name == "career_advisor":
        messages = out.get("messages") or []
        calls = [c for m in messages for c in (m.get("tool_calls") or [])]
        return f"plans {len(calls)} tool calls" if calls else "enough evidence, writes the plan"
    if name == "career_plan" and "career_plan" in out:
        gaps = out["career_plan"]["gaps"]
        return "plan: " + ", ".join(f"{g['skill']} [{g['priority']}]" for g in gaps)
    if name == "write_letter" and "cover_letter" in out:
        return f"draft of {len(out['cover_letter']['body'].split())} words"
    if name == "review_letter" and "review" in out:
        n = len(out["review"]["issues"])
        return "approved" if out["review"]["approved"] else f"{n} issue{'s' * (n > 1)} flagged"
    if name == "verify_claims":
        review = out.get("review")
        if not review:
            return "nothing to dismiss"
        n = len(review["issues"])
        return f"{n} issue{'s' * (n != 1)} left after jevos"
    if name == "human_review":
        return "paused: waiting for a human decision"
    return ""


def tool_detail(name: str, args: dict, out: Any) -> tuple[str, str]:
    out = _json(out.get("content") if isinstance(out, dict) else out)
    if name == "search_skill_taxonomy":
        call = f'"{args.get("query")}"'
        results = (out or {}).get("results") if isinstance(out, dict) else None
        return call, (f"ESCO: {results[0]['preferred_label']}" if results else "ESCO: no concept")
    if name == "check_cv_evidence":
        p = out.get("p_evidence") if isinstance(out, dict) else None
        verdict = "in the CV" if out.get("present") else "not in the CV"
        return f'"{args.get("skill")}"', f"P={p:.2f} · {verdict}"
    if name == "search_learning_resources":
        results = out.get("results") if isinstance(out, dict) else []
        return f'"{args.get("skill")}"', (results[0]["title"] if results else "no resource")
    return json.dumps(args, ensure_ascii=False), ""


def build_record(analysis: dict, observations: list[dict]) -> dict:
    t0 = _ts(observations[0]["startTime"])
    events = []
    for o in observations:
        is_node = o["type"] == "CHAIN" and o["name"] in NODES
        is_tool = o["type"] == "TOOL" and o["name"] in TOOLS
        if not (is_node or is_tool):
            continue
        start = (_ts(o["startTime"]) - t0).total_seconds()
        end = (_ts(o["endTime"]) - t0).total_seconds() if o.get("endTime") else start
        event = {"t": round(start, 1), "duration": round(end - start, 1), "name": o["name"]}
        if is_node:
            event |= {"kind": "node", "detail": node_detail(o["name"], _json(o.get("output")))}
        else:
            call, result = tool_detail(
                o["name"], _json(o.get("input")) or {}, _json(o.get("output"))
            )
            event |= {"kind": "tool", "call": call, "result": result}
        events.append(event)

    match, plan, letter = analysis["match"], analysis["career_plan"], analysis["cover_letter"]
    return {
        "analysis": {
            "status": analysis["status"],
            "offer": f"{analysis['offer']['title']} · {analysis['offer']['company']}",
            "candidate": analysis["profile"]["name"],
            "score": match["score"],
            "matched_skills": match["matched_skills"],
            "gaps": [{"skill": g["skill"], "level": g["level"]} for g in match["gaps"]],
            "plan_summary": plan["summary"] if plan else None,
            "letter_subject": letter["subject"] if letter else None,
            "letter_words": len(letter["body"].split()) if letter else 0,
            "review_issues": len((analysis.get("review") or {}).get("issues") or []),
            "review_iterations": analysis.get("review_iterations"),
        },
        "events": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("analysis_id")
    parser.add_argument("-o", "--output", type=Path, default=Path("docs/demo/run.json"))
    args = parser.parse_args()

    api = os.getenv("JOBMATCH_API_URL", "http://127.0.0.1:8000")
    host = os.getenv("LANGFUSE_HOST", "http://127.0.0.1:3000")
    auth = (
        os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-jobmatch-local"),
        os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-jobmatch-local"),
    )
    analysis = httpx.get(f"{api}/analyses/{args.analysis_id}", timeout=30).raise_for_status().json()
    record = build_record(analysis, fetch_observations(args.analysis_id, host, auth))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{len(record['events'])} events written to {args.output}")


if __name__ == "__main__":
    main()
