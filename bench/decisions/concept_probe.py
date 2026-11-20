"""Probe: is the decision model reliable on BROAD skill concepts, not only concrete skills?

The main benchmark (`run_bench.py`, task `skill_evidence`) covers concrete skills (Python,
Docker, Spark...). The career advisor also asks about broad concepts (ESCO labels or its own
words: "infrastructure as code", "cloud technologies"). This probe measures that case on the
sample data engineer CV, with the exact production question and calibration
(`decisions.service._ask_yes`, task `skill_evidence`).

Labels are a human reading of `data/samples/cv_data_engineer.txt`. Results (jevos-v5,
2026-11-20): 11/13. Both errors are the CATEGORY of a skill the CV lacks:
"infrastructure as code" P=0.90 (Terraform 0.05, Ansible 0.02) and "data streaming" P=0.87
(Kafka 0.03). A stricter question ("does the CV name a specific tool that is a direct form
of ...") was tested and scored 6/13 (it says no to everything, Docker included), so the
production question is kept and the career prompt forbids checking a gap's own category.

    uv run python bench/decisions/concept_probe.py      # needs jevos (make infra)
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from jobmatch.decisions import service

CV = Path(__file__).resolve().parents[2] / "data" / "samples" / "cv_data_engineer.txt"

# skill -> does the CV show it? (True = yes)
CASES: dict[str, bool] = {
    # concrete skills (in the main benchmark's scope)
    "Docker": True,
    "Terraform": False,
    "Kafka": False,
    # broad concepts, present through concrete evidence
    "continuous integration": True,  # GitLab CI
    "containerization": True,  # Docker
    "cloud technologies": True,  # AWS, S3, Redshift
    "data warehouse": True,  # Redshift warehouse modelling
    "data quality": True,  # Great Expectations
    "ETL": True,  # PySpark / Airflow pipelines
    # broad concepts the CV does NOT show
    "infrastructure as code": False,  # no Terraform / Ansible / CloudFormation
    "data streaming": False,  # batch pipelines only
    "machine learning": False,
    "DevOps": False,  # CI + Docker only, no ops / infra role
}


async def main() -> None:
    cv = CV.read_text(encoding="utf-8")
    questions = [service.SKILL_QUESTION.format(s=s) for s in CASES]
    probs = await service._ask_yes("skill_evidence", cv, questions)
    correct = 0
    for (skill, expected), p in zip(CASES.items(), probs, strict=True):
        ok = (p >= 0.5) == expected
        correct += ok
        mark = "" if ok else "   <-- wrong"
        print(f"{skill:26} expected={'yes' if expected else 'no ':3}  P={p:.2f}{mark}")
    print(f"\n{correct}/{len(CASES)} correct")


if __name__ == "__main__":
    asyncio.run(main())
