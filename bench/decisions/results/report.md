# Decision models benchmark

Generated 2026-11-04 15:03. Metrics on the **test** split; 'cal' = after temperature scaling fitted per backend × task on the **calib** split. auto@5% = largest share of test items that can be answered automatically (most confident first) with ≤ 5% errors. See README.md for definitions.

## Dataset

| task | lang | calib | test |
|---|---|---|---|
| claim_supported | fr | 10 | 29 |
| claim_supported | en | 6 | 7 |
| doc_type | fr | 6 | 9 |
| doc_type | en | 0 | 5 |
| seniority | fr | 5 | 10 |
| seniority | en | 2 | 3 |
| skill_evidence | fr | 26 | 58 |
| skill_evidence | en | 5 | 13 |

## skill_evidence

| backend | lang | n | acc | macro-F1 | Brier raw | ECE raw | Brier cal | ECE cal | T | auto@5% raw | auto@5% cal | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| majority-baseline | all | 71 | 0.563 | 0.360 | 0.493 | 0.017 | 0.492 | 0.015 | 0.97 | 0.00 | 0.00 | – | – |
| majority-baseline | fr | 58 | 0.586 | 0.370 | 0.488 | 0.039 | 0.488 | 0.038 | 0.97 | 0.00 | 0.00 | – | – |
| majority-baseline | en | 13 | 0.462 | 0.316 | 0.512 | 0.085 | 0.512 | 0.087 | 0.97 | 0.00 | 0.00 | – | – |
| jevos | all | 71 | 0.901 | 0.901 | 0.160 | 0.075 | 0.154 | 0.034 | 0.86 | 0.86 | 0.86 | 50 | 89 |
| jevos | fr | 58 | 0.879 | 0.878 | 0.183 | 0.084 | 0.179 | 0.030 | 0.86 | 0.78 | 0.78 | 50 | 90 |
| jevos | en | 13 | 1.000 | 1.000 | 0.055 | 0.140 | 0.041 | 0.113 | 0.86 | 1.00 | 1.00 | 40 | 76 |
| jevos@q-en | all | 71 | 0.944 | 0.943 | 0.129 | 0.084 | 0.117 | 0.056 | 0.74 | 0.94 | 0.94 | 45 | 72 |
| jevos@q-en | fr | 58 | 0.931 | 0.930 | 0.145 | 0.072 | 0.137 | 0.069 | 0.74 | 0.88 | 0.88 | 46 | 72 |
| jevos@q-en | en | 13 | 1.000 | 1.000 | 0.055 | 0.140 | 0.030 | 0.089 | 0.74 | 1.00 | 1.00 | 41 | 72 |

## claim_supported

| backend | lang | n | acc | macro-F1 | Brier raw | ECE raw | Brier cal | ECE cal | T | auto@5% raw | auto@5% cal | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| majority-baseline | all | 36 | 0.472 | 0.321 | 0.500 | 0.028 | 0.500 | 0.028 | 20.00 | 0.00 | 0.00 | – | – |
| majority-baseline | fr | 29 | 0.414 | 0.293 | 0.500 | 0.086 | 0.500 | 0.086 | 20.00 | 0.00 | 0.00 | – | – |
| majority-baseline | en | 7 | 0.714 | 0.417 | 0.500 | 0.214 | 0.500 | 0.214 | 20.00 | 0.00 | 0.00 | – | – |
| jevos | all | 36 | 0.528 | 0.429 | 0.666 | 0.349 | 0.567 | 0.280 | 1.49 | 0.25 | 0.25 | 61 | 175 |
| jevos | fr | 29 | 0.414 | 0.293 | 0.822 | 0.452 | 0.691 | 0.373 | 1.49 | 0.17 | 0.17 | 61 | 175 |
| jevos | en | 7 | 1.000 | 1.000 | 0.020 | 0.077 | 0.054 | 0.148 | 1.49 | 1.00 | 1.00 | 36 | 138 |
| jevos@q-en | all | 36 | 1.000 | 1.000 | 0.024 | 0.085 | 0.017 | 0.067 | 0.89 | 1.00 | 1.00 | 54 | 164 |
| jevos@q-en | fr | 29 | 1.000 | 1.000 | 0.024 | 0.087 | 0.018 | 0.069 | 0.89 | 1.00 | 1.00 | 56 | 165 |
| jevos@q-en | en | 7 | 1.000 | 1.000 | 0.020 | 0.077 | 0.014 | 0.060 | 0.89 | 1.00 | 1.00 | 32 | 140 |

## seniority

| backend | lang | n | acc | macro-F1 | Brier raw | ECE raw | Brier cal | ECE cal | T | auto@5% raw | auto@5% cal | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| majority-baseline | all | 13 | 0.231 | 0.094 | 0.755 | 0.047 | 0.760 | 0.055 | 0.74 | 0.00 | 0.00 | – | – |
| majority-baseline | fr | 10 | 0.300 | 0.115 | 0.726 | 0.022 | 0.722 | 0.014 | 0.74 | 0.00 | 0.00 | – | – |
| majority-baseline | en | 3 | 0.000 | 0.000 | 0.852 | 0.278 | 0.884 | 0.286 | 0.74 | 0.00 | 0.00 | – | – |
| jevos | all | 13 | 0.462 | 0.499 | 0.784 | 0.385 | 0.692 | 0.208 | 1.76 | 0.00 | 0.00 | 106 | 144 |
| jevos | fr | 10 | 0.400 | 0.350 | 0.825 | 0.535 | 0.708 | 0.281 | 1.76 | 0.00 | 0.00 | 109 | 146 |
| jevos | en | 3 | 0.667 | 0.556 | 0.650 | 0.169 | 0.636 | 0.494 | 1.76 | 0.67 | 0.67 | 86 | 131 |
| jevos@q-en | all | 13 | 0.538 | 0.563 | 0.672 | 0.222 | 0.678 | 0.277 | 0.94 | 0.00 | 0.00 | 82 | 140 |
| jevos@q-en | fr | 10 | 0.500 | 0.548 | 0.678 | 0.238 | 0.685 | 0.312 | 0.94 | 0.00 | 0.00 | 81 | 141 |
| jevos@q-en | en | 3 | 0.667 | 0.556 | 0.650 | 0.169 | 0.653 | 0.160 | 0.94 | 0.67 | 0.67 | 90 | 129 |

## doc_type

| backend | lang | n | acc | macro-F1 | Brier raw | ECE raw | Brier cal | ECE cal | T | auto@5% raw | auto@5% cal | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| majority-baseline | all | 14 | 0.286 | 0.148 | 0.667 | 0.048 | 0.667 | 0.048 | 20.00 | 0.00 | 0.00 | – | – |
| majority-baseline | fr | 9 | 0.333 | 0.167 | 0.667 | 0.000 | 0.667 | 0.000 | 20.00 | 0.00 | 0.00 | – | – |
| majority-baseline | en | 5 | 0.200 | 0.111 | 0.667 | 0.133 | 0.667 | 0.133 | 20.00 | 0.00 | 0.00 | – | – |
| jevos | all | 14 | 0.429 | 0.344 | 0.691 | 0.400 | 0.608 | 0.304 | 1.95 | 0.07 | 0.07 | 92 | 190 |
| jevos | fr | 9 | 0.333 | 0.281 | 0.837 | 0.514 | 0.692 | 0.442 | 1.95 | 0.11 | 0.11 | 88 | 191 |
| jevos | en | 5 | 0.600 | 0.444 | 0.429 | 0.229 | 0.457 | 0.096 | 1.95 | 0.40 | 0.40 | 97 | 173 |
| jevos@q-en | all | 14 | 0.500 | 0.413 | 0.643 | 0.300 | 0.650 | 0.147 | 20.00 | 0.00 | 0.00 | 85 | 189 |
| jevos@q-en | fr | 9 | 0.444 | 0.389 | 0.762 | 0.358 | 0.657 | 0.092 | 20.00 | 0.00 | 0.00 | 75 | 189 |
| jevos@q-en | en | 5 | 0.600 | 0.444 | 0.429 | 0.229 | 0.637 | 0.247 | 20.00 | 0.40 | 0.20 | 91 | 168 |

## Resources

| backend | load s | run s (cached run) | conc. | RSS after load MB | peak RSS MB | children peak RSS MB | errors | missing |
|---|---|---|---|---|---|---|---|---|
| majority-baseline | – | – | – | – | – | – | 0 | 0 |
| jevos | 2.1 | 7 | 2 | 47 | 48 | 1558 | 0 | 0 |
| jevos@q-en | 2.1 | 7 | 2 | 48 | 48 | 1546 | 0 | 0 |

RSS = the benchmark worker process (Python + client). Children = processes the backend spawned (e.g. `jev serve`): RSS after load and peak (VmHWM). A server that was already running before the benchmark is not measured. Qwen (if run) lives in llama-server on the GPU: not measured.

## Verdict

Bar for 'very good': on each yes/no task (skill_evidence, claim_supported) and EACH language, test accuracy ≥ 0.85 and automatable share at 5% error (after calibration) ≥ 0.5. Choice tasks are reported for information.

| backend | task | lang | n | acc | acc 95% CI low (Wilson) | auto@5% cal | pass |
|---|---|---|---|---|---|---|---|
| jevos | skill_evidence | fr | 58 | 0.879 | 0.77 | 0.78 | ✅ |
| jevos | skill_evidence | en | 13 | 1.000 | 0.77 | 1.00 | ✅ |
| jevos | claim_supported | fr | 29 | 0.414 | 0.26 | 0.17 | ❌ |
| jevos | claim_supported | en | 7 | 1.000 | 0.65 | 1.00 | ✅ |
| jevos | seniority | fr | 10 | 0.400 | 0.17 | 0.00 | (❌) |
| jevos | seniority | en | 3 | 0.667 | 0.21 | 0.67 | (❌) |
| jevos | doc_type | fr | 9 | 0.333 | 0.12 | 0.11 | (❌) |
| jevos | doc_type | en | 5 | 0.600 | 0.23 | 0.40 | (❌) |
| jevos@q-en | skill_evidence | fr | 58 | 0.931 | 0.84 | 0.88 | ✅ |
| jevos@q-en | skill_evidence | en | 13 | 1.000 | 0.77 | 1.00 | ✅ |
| jevos@q-en | claim_supported | fr | 29 | 1.000 | 0.88 | 1.00 | ✅ |
| jevos@q-en | claim_supported | en | 7 | 1.000 | 0.65 | 1.00 | ✅ |
| jevos@q-en | seniority | fr | 10 | 0.500 | 0.24 | 0.00 | (❌) |
| jevos@q-en | seniority | en | 3 | 0.667 | 0.21 | 0.67 | (❌) |
| jevos@q-en | doc_type | fr | 9 | 0.444 | 0.19 | 0.00 | (❌) |
| jevos@q-en | doc_type | en | 5 | 0.600 | 0.23 | 0.20 | (❌) |

- **jevos: NOT 'very good' in both FR and EN.** Below the bar: claim_supported/fr (acc 0.41, auto 0.17).
- **jevos@q-en: meets the bar in BOTH French and English** on yes/no tasks.
- `jevos` fails but `jevos@q-en` passes: the weakness is the QUESTION language, not the document language — French documents work when the question and option descriptions are written in English (which the graph controls).
- English test sets are tiny (n ≤ 13 per task): a perfect score on 7 items still has a 95% lower bound around 0.65 — see the Wilson column before trusting 'EN passes'.

## jevos: 10 most confident errors (all splits, for label spot-check)

| id | task | lang | split | label | predicted | confidence | label note |
|---|---|---|---|---|---|---|---|
| doc-008-cv-accountant | doc_type | fr | test | cv | other | 0.970 |  |
| skill-077-lead | skill_evidence | fr | test | yes | no | 0.955 | Hard synonym: deploys on OpenShift, a Kubernetes distribution. |
| skill-040-ml | skill_evidence | fr | test | yes | no | 0.938 | Hard: BigQuery is a Google Cloud product used in the job. |
| claim-037-frontend | claim_supported | fr | test | yes | no | 0.929 |  |
| claim-045-lead | claim_supported | fr | test | yes | no | 0.924 |  |
| claim-019-nurse | claim_supported | fr | calib | yes | no | 0.913 |  |
| claim-043-logistics | claim_supported | fr | test | yes | no | 0.906 |  |
| claim-025-data_engineer | claim_supported | fr | test | yes | no | 0.906 |  |
| skill-061-data_engineer | skill_evidence | fr | calib | yes | no | 0.897 | Power BI dashboards for clients. |
| claim-030-data_engineer | claim_supported | fr | test | yes | no | 0.896 | CV profile states 4 years of experience (2020-2024, incl. BI role and internship). |

## jevos@q-en: 10 most confident errors (all splits, for label spot-check)

| id | task | lang | split | label | predicted | confidence | label note |
|---|---|---|---|---|---|---|---|
| doc-008-cv-accountant | doc_type | fr | test | cv | other | 0.951 |  |
| skill-077-lead | skill_evidence | fr | test | yes | no | 0.929 | Hard synonym: deploys on OpenShift, a Kubernetes distribution. |
| skill-040-ml | skill_evidence | fr | test | yes | no | 0.928 | Hard: BigQuery is a Google Cloud product used in the job. |
| claim-042-accountant | claim_supported | fr | calib | no | yes | 0.887 | Inflated duration: 5y10m in the cabinet. |
| doc-007-cv-support | doc_type | fr | test | cv | other | 0.862 |  |
| skill-061-data_engineer | skill_evidence | fr | calib | yes | no | 0.857 | Power BI dashboards for clients. |
| skill-068-bank | skill_evidence | fr | calib | no | yes | 0.826 | Only a beginner MOOC still in progress. |
| sen-007-bank | seniority | fr | test | senior | mid | 0.758 | Hard: 5y6m analyst + 3y adviser (~8.5 y in banking), no reports. |
| sen-013-s_mid_fr | seniority | fr | test | mid | senior | 0.692 | ~4.5 years (2020-) + 2y alternance. |
| doc-011-cv-qa | doc_type | en | test | cv | other | 0.691 |  |

## Observations (auto-generated)

- **skill_evidence**: best macro-F1 jevos@q-en (0.943, acc 0.944); highest automatable share at 5% error jevos@q-en (94% of test items).
  - majority-baseline: FR/EN accuracy gap 0.59 vs 0.46 (EN n=13, small).
  - jevos: FR/EN accuracy gap 0.88 vs 1.00 (EN n=13, small).
- **claim_supported**: best macro-F1 jevos@q-en (1.000, acc 1.000); highest automatable share at 5% error jevos@q-en (100% of test items).
  - jevos: temperature scaling cuts ECE 0.349 → 0.280 (T=1.49).
  - majority-baseline: FR/EN accuracy gap 0.41 vs 0.71 (EN n=7, small).
  - jevos: FR/EN accuracy gap 0.41 vs 1.00 (EN n=7, small).
- **seniority**: best macro-F1 jevos@q-en (0.563, acc 0.538); highest automatable share at 5% error majority-baseline (0% of test items).
  - jevos: temperature scaling cuts ECE 0.385 → 0.208 (T=1.76).
  - jevos@q-en: calibration fitted on the small calib split HURTS on test (ECE 0.222 → 0.277).
  - majority-baseline: FR/EN accuracy gap 0.30 vs 0.00 (EN n=3, small).
  - jevos: FR/EN accuracy gap 0.40 vs 0.67 (EN n=3, small).
  - jevos@q-en: FR/EN accuracy gap 0.50 vs 0.67 (EN n=3, small).
- **doc_type**: best macro-F1 jevos@q-en (0.413, acc 0.500); highest automatable share at 5% error jevos (7% of test items).
  - jevos: temperature scaling cuts ECE 0.400 → 0.304 (T=1.95).
  - jevos@q-en: temperature scaling cuts ECE 0.300 → 0.147 (T=20.00).
  - majority-baseline: FR/EN accuracy gap 0.33 vs 0.20 (EN n=5, small).
  - jevos: FR/EN accuracy gap 0.33 vs 0.60 (EN n=5, small).
  - jevos@q-en: FR/EN accuracy gap 0.44 vs 0.60 (EN n=5, small).
- Test splits are small (EN especially): differences of a few points are within noise.
