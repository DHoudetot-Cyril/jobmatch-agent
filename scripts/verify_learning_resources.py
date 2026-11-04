"""Verify every URL of data/career/learning_resources.json; drop dead ones, stamp `verified_at`.

ONE-OFF maintenance script (network). The catalog itself is curated by hand (official docs,
official free tutorials/courses, well-known free courses, official certification pages).
A URL is kept if HEAD (or GET as a fallback, redirects followed) ends with a 2xx/3xx status.

    uv run python scripts/verify_learning_resources.py [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
from pathlib import Path

import httpx

PATH = Path(__file__).resolve().parents[1] / "data" / "career" / "learning_resources.json"
UA = "Mozilla/5.0 (X11; Linux x86_64) jobmatch-agent link checker"


async def check(client: httpx.AsyncClient, sem: asyncio.Semaphore, url: str) -> int | None:
    async with sem:
        for method in ("HEAD", "GET"):
            try:
                r = await client.request(method, url)
                if r.status_code < 400:
                    return r.status_code
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.2)
        return r.status_code if "r" in locals() else None


async def main(dry_run: bool) -> None:
    data = json.loads(PATH.read_text())
    rows = data["resources"]
    sem = asyncio.Semaphore(4)
    async with httpx.AsyncClient(
        follow_redirects=True, timeout=20, headers={"User-Agent": UA}
    ) as client:
        statuses = await asyncio.gather(*(check(client, sem, r["url"]) for r in rows))
    today = dt.date.today().isoformat()
    kept, rejected = [], []
    for row, status in zip(rows, statuses, strict=True):
        if status is not None and status < 400:
            kept.append({**row, "verified_at": today})
        else:
            rejected.append((status, row["url"]))
    for status, url in rejected:
        print(f"REJECTED {status} {url}")
    print(f"kept {len(kept)} / {len(rows)}")
    if not dry_run:
        data["resources"] = kept
        PATH.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    asyncio.run(main(ap.parse_args().dry_run))
