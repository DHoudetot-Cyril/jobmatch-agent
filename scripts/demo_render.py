"""Render docs/demo.gif from a real run recorded by scripts/demo_extract.py.

Every line, number and timestamp comes from docs/demo/run.json (the API result and the Langfuse
trace of one real analysis). Only the pace changes: the ~9 min run is replayed in ~45 s, and the
clock in the corner shows the real elapsed time.

    uv run python scripts/demo_render.py            # -> docs/demo.gif
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1040, 720
PAD_X, TOP, LINE_H = 22, 50, 20
COLS = 118
MAX_LINES = (H - TOP - 46) // LINE_H

BG, BAR, FG, DIM = (24, 24, 27), (39, 39, 42), (228, 228, 231), (130, 130, 140)
ACCENT, TOOL, OK, WARN, TEAM = (
    (217, 119, 87),
    (125, 180, 255),
    (134, 239, 172),
    (250, 204, 21),
    (196, 181, 253),
)

FONT_DIR = "/usr/share/fonts/truetype/dejavu/"


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    for path in (FONT_DIR + name, name):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    raise SystemExit("DejaVu Sans Mono not found (apt install fonts-dejavu-core).")


FONT, BOLD = _font("DejaVuSansMono.ttf", 14), _font("DejaVuSansMono-Bold.ttf", 14)
SMALL = _font("DejaVuSansMono.ttf", 12)

TEAMS = {
    "parse_offer": "analysis team",
    "parse_cv": "analysis team",
    "score_match": "matching team",
    "career_advisor": "career team · autonomous ReAct agent, 3 local tools",
    "write_letter": "writing team · writer / reviewer loop",
    "human_review": "human in the loop",
}
SPINNER = "◐◓◑◒"


def clock(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


class Screen:
    def __init__(self, title: str) -> None:
        self.title = title
        self.lines: list[list[tuple[str, tuple, ImageFont.FreeTypeFont]]] = []
        self.status: list[tuple[str, tuple]] | None = None
        self.real_t = 0.0
        self.frames: list[Image.Image] = []
        self.durations: list[int] = []

    def add(self, *segments) -> None:
        self.lines.append([(s[0], s[1], s[2] if len(s) > 2 else FONT) for s in segments])

    def snap(self, ms: int) -> None:
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, W, 32], fill=BAR)
        for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
            d.ellipse([14 + i * 21, 10, 26 + i * 21, 22], fill=c)
        d.text((W // 2, 16), self.title, font=SMALL, fill=DIM, anchor="mm")
        d.text(
            (W - 16, 16), f"real time {clock(self.real_t)}", font=SMALL, fill=ACCENT, anchor="rm"
        )
        y = TOP
        for segments in self.lines[-MAX_LINES:]:
            x = PAD_X
            for text, color, font in segments:
                d.text((x, y), text, font=font, fill=color)
                x += d.textlength(text, font=font)
            y += LINE_H
        if self.status:
            x = PAD_X
            for text, color in self.status:
                d.text((x, H - 44), text, font=FONT, fill=color)
                x += d.textlength(text, font=FONT)
        d.text(
            (PAD_X, H - 20),
            "Real run (local Qwen 27B on llama.cpp + jevos), "
            "replayed from the API result and the Langfuse trace",
            font=SMALL,
            fill=DIM,
        )
        self.frames.append(img)
        self.durations.append(ms)


def wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width) or [""]


def render(record: dict, output: Path) -> None:
    a = record["analysis"]
    s = Screen("jobmatch-agent — POST /analyses")

    # 1. Submission
    cmd = (
        "curl -X POST :8000/analyses -F cv=@cv_data_engineer.txt -F offer=@offer_data_engineer.txt"
    )
    s.add(("$ ", ACCENT, BOLD), ("", FG))
    s.snap(500)
    for i in [*range(0, len(cmd), 4), len(cmd)]:
        s.lines[-1] = [("$ ", ACCENT, BOLD), (cmd[:i], FG, FONT)]
        s.snap(30)
    s.add(('{"status": "running"}', DIM))
    s.add(("", FG))
    s.snap(700)

    # 2. The graph, node by node, at the real timestamps (compressed)
    events = [e for e in record["events"] if e["name"] not in {"supervisor", "career_tools"}]
    team = None
    for e in events:
        if e["kind"] == "node" and TEAMS.get(e["name"]) and TEAMS[e["name"]] != team:
            team = TEAMS[e["name"]]
            s.add(("── " + team + " " + "─" * max(0, 60 - len(team)), TEAM))
        if e["kind"] == "node":
            # spinner while the node runs: a few frames, the clock advancing in real time
            steps = 1 if e["duration"] < 3 else 4
            for k in range(steps):
                s.real_t = e["t"] + e["duration"] * (k / steps)
                s.status = [
                    (SPINNER[k % len(SPINNER)] + " ", ACCENT),
                    (e["name"], FG),
                    (" running…", DIM),
                ]
                s.snap(min(650, 120 + int(e["duration"] * 6)))
            s.real_t = e["t"] + e["duration"]
            s.status = None
            color = WARN if "issue" in e["detail"] and "left" in e["detail"] else OK
            if e["name"] == "human_review":
                color = WARN
            label = f"{e['name']:<15}"
            first, *rest = wrap(e["detail"], COLS - 30)
            s.add(
                ("  ✓ ", OK), (label, FG, BOLD), (f"{e['duration']:>5.1f}s  ", DIM), (first, color)
            )
            for line in rest:
                s.add((" " * 29, FG), (line, color))
            s.snap(260)
        else:
            s.real_t = e["t"]
            s.add(
                ("      ↳ ", DIM),
                (f"{e['name']}({e['call']})", TOOL),
                ("  →  ", DIM),
                (e["result"], FG),
            )
            s.snap(150)
    s.add(("", FG))
    s.snap(900)

    # 3. Result card
    s.lines = []
    s.title = "jobmatch-agent — GET /analyses/{id}"
    s.add(("$ ", ACCENT, BOLD), ("curl :8000/analyses/{id}", FG))
    s.add(("", FG))
    bar = "█" * (a["score"] // 4) + "░" * (25 - a["score"] // 4)
    s.add(("  Offer      ", DIM), (a["offer"], FG, BOLD))
    s.add(("  Candidate  ", DIM), (a["candidate"], FG))
    s.add(
        ("  Score      ", DIM),
        (bar + "  ", OK),
        (f"{a['score']}/100", OK, BOLD),
        ("  (computed in code)", DIM),
    )
    s.add(("  Matched    ", DIM), (", ".join(a["matched_skills"]), FG))
    gaps = ", ".join(f"{g['skill']} ({g['level'].replace('_', '-')})" for g in a["gaps"])
    s.add(("  Gaps       ", DIM), (gaps, WARN))
    s.add(("", FG))
    s.snap(1200)
    s.add(("  Career plan", DIM))
    for line in wrap(a["plan_summary"] or "", COLS - 6):
        s.add(("    ", FG), (line, FG))
        s.snap(220)
    s.add(("", FG))
    s.add(("  Letter     ", DIM), (f"« {a['letter_subject']} » · {a['letter_words']} words", FG))
    s.add(
        ("  Review     ", DIM),
        (
            f"{a['review_iterations']} writer/reviewer rounds · "
            f"{a['review_issues']} issues left for the human",
            WARN,
        ),
    )
    s.add(
        ("  Status     ", DIM),
        (a["status"], ACCENT, BOLD),
        ("  → POST /analyses/{id}/decision", DIM),
    )
    s.snap(6500)

    # One shared palette, built from a log frame and the result card so every colour is in it.
    sample = Image.new("RGB", (W, 2 * H))
    sample.paste(s.frames[len(s.frames) * 2 // 3], (0, 0))
    sample.paste(s.frames[-1], (0, H))
    palette = sample.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in s.frames]
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output, save_all=True, append_images=frames[1:], duration=s.durations, loop=0, optimize=True
    )
    print(
        f"{output}: {len(frames)} frames, {sum(s.durations) / 1000:.0f} s, "
        f"{output.stat().st_size / 1e6:.1f} MB"
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", type=Path, default=root / "docs/demo/run.json")
    parser.add_argument("-o", "--output", type=Path, default=root / "docs/demo.gif")
    args = parser.parse_args()
    render(json.loads(args.record.read_text(encoding="utf-8")), args.output)


if __name__ == "__main__":
    main()
