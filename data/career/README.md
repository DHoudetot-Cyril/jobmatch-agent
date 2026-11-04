# data/career — local data of the career advisor

Read OFFLINE at runtime by `src/jobmatch/career/tools.py` (directory = setting
`career_data_dir`, env `CAREER_DATA_DIR`, default `data/career`, resolved against the working
directory then the repository root). The network is used only by the two one-off build
scripts below.

| File | Content | Size |
|---|---|---|
| `esco_skills.json` | subset of the ESCO skills taxonomy (272 concepts) | ~0.3 MB |
| `learning_resources.json` | curated, link-checked learning resources (107 entries, 76 skills) | ~40 KB |

## esco_skills.json — ESCO snapshot

**Source & licence.** ESCO (European Skills, Competences, Qualifications and Occupations),
<https://esco.ec.europa.eu>, © European Union, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Data fetched from the public ESCO
API (<https://ec.europa.eu/esco/api>) on the date in `built_at`. Changes: subset only;
alternative labels, descriptions and relations trimmed; fields renamed.

**Content.** One entry per concept: `uri`, `preferred_label {en, fr}`, `alt_labels {en, fr}`
(≤ 8 each), `description` (English, ≤ 400 chars), `skill_group` (ESCO hierarchy parent) and
`related` (≤ 12 `{uri, label, relation}`) taken from the real ESCO skill-to-skill relations
`broaderSkill`, `narrowerSkill`, `hasEssentialSkill`, `isEssentialForSkill`,
`hasOptionalSkill`, `isOptionalForSkill`. Also `seeds` (seed → kept URIs, the first one is
used as an extra search alias), `seeds_not_found` and `curated_aliases`.

**How it is built** (`scripts/build_esco_snapshot.py`): 273 seeds (every skill of
`data/samples` and `evals/dataset` + common tech / data / devops / support / accounting /
logistics / healthcare / management skills, ~20 of them in French). Each seed is searched
(`/search?type=skill`, in its language); a hit is kept only if its labels really match the
seed (one-word seeds must match a *preferred* label: alternative labels are full of
homonyms, e.g. "Excel" → "demand performers excel"); every concept is fetched once
(`/resource/skill`). Concurrency 3, 0.15 s sleep, retries with backoff; raw responses cached
in `~/.cache/jobmatch/esco` (outside the repo; `ESCO_CACHE_DIR` to override).

```bash
uv run python scripts/build_esco_snapshot.py            # reuse the cache (~2 min cold)
uv run python scripts/build_esco_snapshot.py --no-cache
```

**Stats (2026-11-04):** 273 seeds, 193 resolved (31 of them through `curated_aliases`),
272 concepts.

**Curated aliases** (hand-made approximations, ESCO has no exact concept): Excel / tableur
→ *use spreadsheets software*; Office 365 → *use microsoft office*; Bash / shell scripting →
*use scripting programming*; help desk / support informatique / technical support →
*handle helpdesk problems*; system administration → *administer ICT system*; CACES /
cariste / forklift → *operate forklift*; order picking / préparation de commandes → *pick
orders for dispatching*; WMS → *manage a warehouse management system*; AWS / Azure / Google
Cloud → *cloud technologies*; Git → *tools for software configuration management*; team
management / gestion d'équipe / encadrement → *lead a team*; comptabilité générale →
*accounting*; Power BI / Tableau → *business intelligence*.

**Not found in ESCO** (no matching concept; the tool then answers "no ESCO concept" and the
advisor should use a broader term): AWS Glue, Amazon Redshift, Amazon S3, Angular, Apache
Airflow, Apache Hive, Apache Spark, CI/CD, Celery, Cypress, DNS, Django, Docker, FastAPI,
Flask, GLPI, GitHub Actions, GitLab CI, Grafana, GraphQL, HTML, Jest, Kafka, Kanban, Kotlin,
Kubernetes, Linux, MLflow, MongoDB, Next.js, Node.js, Pennylane, Prometheus, PyTorch, REST
API, Redis, Sage, Scrum, Selenium, Spring framework, TensorFlow, Terraform, UX design,
VMware, Vue.js, XML, Zabbix, dbt, pandas, pytest, scikit-learn, containerisation,
infrastructure as code, microservices, deep learning, computer vision, time series
forecasting, cybersecurity, déclaration de TVA, sécurité au travail, … (full list in
`seeds_not_found`). ESCO covers few vendor tools: Python, SQL, Ansible, Jenkins, JavaScript,
PostgreSQL, SAP R3, ITIL (*apply operations for an ITIL-based environment*) are in it;
Kubernetes, Docker, Terraform and Kafka are not.

## learning_resources.json — learning catalog

Hand-curated: official documentation, official free tutorials/courses (Kubernetes, HashiCorp,
dbt Labs, Confluent, AWS/Microsoft/SAP learning, MDN, Kaggle Learn, Google MLCC, Inria
scikit-learn MOOC, fast.ai, CS50, DataTalks.Club zoomcamps…), official certification pages
(CKA, AWS, Terraform Associate, ITIL 4, ISTQB, PL-300, MS-900) and official French public
information pages (INRS for CACES, impots.gouv.fr / service-public for TVA, ANC, net-entreprises
for DSN/paie, ANSSI SecNumacadémie, RGAA, Mon Compte Formation). Titles/descriptions are
ours; no third-party content is copied.

Fields: `skill` (canonical), `aliases`, `title`, `url`, `provider`, `kind`
(docs/course/tutorial/certification), `free`, `level`, `language`, `est_hours` (only when the
provider states it — Kaggle Learn `estimatedTimeHours` — else null), `verified_at`.

**Link check** (`scripts/verify_learning_resources.py`): HEAD then GET fallback, redirects
followed, kept only if the final status is 2xx/3xx; rejected entries are dropped and
`verified_at` is stamped. First pass on 108 candidates: 7 rejected (404: two INRS pages,
ANC and impots.gouv.fr old paths; 403 bot-blocking: sap.com, sage.com; 429: docs.ansible.com)
→ 4 replaced by verified pages, 3 dropped; redirected URLs were replaced by their final URL
and one soft-404 (AWS Skill Builder course redirecting to a "not found" search) was replaced.
Final pass: 107/107 OK.

```bash
uv run python scripts/verify_learning_resources.py --dry-run   # report only
uv run python scripts/verify_learning_resources.py             # drop dead links, stamp date
```
