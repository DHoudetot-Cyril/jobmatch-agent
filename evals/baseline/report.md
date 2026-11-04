# JobMatch evaluation report

Generated 2026-11-04T14:06:06+00:00 · judge: qwen (local) · cases: 8 · judging 403s

| case | score (expected) | in range | skills recall | gaps recall | faithfulness | relevance | gap usefulness |
|---|---|---|---|---|---|---|---|
| strong_fit_backend | 94 (75-100) | 1.00 | 1.00 | 1.00 | 0.90 | 1.00 | 1.00 |
| partial_fit_devops | 40 (30-65) | 1.00 | 1.00 | 1.00 | 1.00 | 0.60 | 0.90 |
| mismatch_nurse_data | 8 (0-25) | 1.00 | 1.00 | 1.00 | 1.00 | 0.30 | 0.90 |
| junior_vs_senior_frontend | 48 (30-65) | 1.00 | 1.00 | 1.00 | 0.90 | 0.40 | 0.90 |
| overqualified_support | 96 (65-100) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| english_offer_ml | 58 (50-85) | 1.00 | 1.00 | 1.00 | 0.90 | 0.70 | 1.00 |
| employment_gap_accountant | 71 (60-95) | 1.00 | 1.00 | 1.00 | 0.90 | 0.90 | 0.90 |
| few_skills_logistics | 61 (40-80) | 1.00 | 1.00 | 1.00 | 0.80 | 0.80 | 0.90 |
| **average** | | **1.00** | **1.00** | **1.00** | **0.93** | **0.71** | **0.94** |

## Thresholds

- `letter_faithfulness` average 0.93 (>= 0.7): **ok**
- `score_in_range` average 1.00 (>= 0.6): **ok**

## Details

### strong_fit_backend

_Strong fit — mid-level Python backend developer for a matching backend offer._

- graph run: 356.4s, review iterations 2, reviewer approved: True
- matched skills: Python, Django, Django REST Framework, PostgreSQL, Docker, GitHub Actions, pytest, Celery, Redis
- gaps: Kubernetes
- letter language: fr
- **score_in_range** 1.00 — score 94, expected [75, 100]
- **skills_recall** 1.00 — all found
- **gaps_recall** 1.00 — all flagged
- **letter_faithfulness** 0.90 — The cover letter accurately reflects the CV for all major candidate claims: Parcelia employer, Django REST Framework APIs handling 40,000 orders/day, PostgreSQL query optimization reducing average response time by a factor of three, Celery/Redis asynchronous tasks for emails and invoices, pytest coverage from 45% to 82%, GitHub Actions pipelines, Docker images, and Docker/Linux/Git/CI practices. The only weak point is the statement about not yet having Kubernetes experience, which is not explicitly stated in the CV, though it is a negative claim rather than an invented positive credential.
- **letter_relevance** 1.00 — The response is in French, matching the expected language, and follows a clear cover-letter structure with greeting, body, and closing. It explicitly links the candidate’s experience to key job requirements, including API REST development with Django REST Framework, PostgreSQL optimization, pytest, Docker, GitHub Actions, Celery/Redis, and willingness to participate in on-call rotations. The tone is professional and concise, with specific examples tied to the role rather than generic claims.
- **gap_usefulness** 1.00 — The advice directly addresses the only identified gap, Kubernetes, with a concrete and realistic plan for a nice-to-have skill: a specific short course (KodeKloud Kubernetes Fundamentals), a hands-on project using the candidate's existing Django/Docker/Linux/CI background (deploying the API on kind/minikube with Service, Ingress, and HPA), and a clear way to present it in an interview. It is specific, actionable, and grounded in the candidate's CV.

### partial_fit_devops

_Partial fit — Linux sysadmin applying to a cloud DevOps engineer offer._

- graph run: 429.5s, review iterations 2, reviewer approved: False
- matched skills: Ansible, Linux, Docker, Bash, Python
- gaps: AWS, Terraform, Kubernetes, Helm, Prometheus, Grafana, AWS, CKA, GitLab CI
- letter language: fr
- **score_in_range** 1.00 — score 40, expected [30, 65]
- **skills_recall** 1.00 — all found
- **gaps_recall** 1.00 — all flagged
- **letter_faithfulness** 1.00 — All factual claims about the candidate are supported by the CV: Clinique des Pins, 120 Linux servers (Debian and Rocky Linux), Ansible with 60 roles, Docker Compose for a wiki and internal tools, Bash scripts for backups and log rotation, Zabbix supervision, and skills in Linux, Bash, Ansible, Docker, and Python notions. No invented employers, figures, certifications, or inflated experience are present.
- **letter_relevance** 0.60 — The letter is in French, professionally structured, and concisely links concrete experience to several requirements (Linux administration, Ansible roles, Docker, Bash, monitoring). However, it does not substantively address the indispensable Kubernetes and Terraform requirements with relevant experience, only acknowledging gaps and willingness to learn, which weakens alignment with the core job requirements.
- **gap_usefulness** 0.90 — The response gives concrete, actionable advice for each listed gap, naming specific resources or projects (HashiCorp Terraform course, kind/k3s cluster, Helm chart, Prometheus + node_exporter, Grafana dashboards, GitLab CI pipeline) and consistently leveraging Nadia's existing Linux, Ansible, Zabbix, and Docker Compose experience. Minor shortcomings are the repeated AWS advice for the duplicated gap and the somewhat heavy CKA recommendation for a nice-to-have, though it remains relevant and not unrealistic.

### mismatch_nurse_data

_Clear mismatch — a nurse applying to a data scientist offer (different field)._

- graph run: 454.7s, review iterations 2, reviewer approved: False
- matched skills: Hospital domain
- gaps: Python, SQL, pandas, scikit-learn, Machine Learning, Regression, Decision Trees, Cross-validation, Statistics, PyTorch
- letter language: fr
- **score_in_range** 1.00 — score 8, expected [0, 25]
- **skills_recall** 1.00 — all found
- **gaps_recall** 1.00 — all flagged
- **letter_faithfulness** 1.00 — The cover letter’s factual claims are well supported by the CV: it correctly states she is an Infirmière diplômée d’État, worked in réanimation at Hôpital Saint-Aubin with 12 critical-care beds, supervised nursing students, participated in implementing a patient record software and trained colleagues, and has Excel experience. It also accurately references her réanimation and médecine polyvalente experience and multidisciplinary teamwork. The data-science technologies are framed as skills she wants to learn rather than claimed experience, so no clear hallucination or contradiction is present.
- **letter_relevance** 0.30 — The letter is in French, professional, and has a clear cover-letter structure. It effectively uses the candidate's hospital experience to address the 'hospital environment' plus and communication with medical teams. However, it does not explicitly link the candidate's experience to the core data science requirements, such as Python/pandas/scikit-learn, SQL, ML modeling, or MLOps industrialization; it mainly states intent to learn these skills, so alignment with the key job requirements is weak.
- **gap_usefulness** 0.90 — Each gap receives actionable advice with named resources (Kaggle Learn, SQLBolt, Mode Analytics, scikit-learn guide) and hospital-domain projects, while leveraging the candidate's nursing, ICU, EHR, and lab-result background. Minor shortcomings are that some recommendations remain somewhat generic ('short course', 'practical course') and the PyTorch project is ambitious for a nice-to-have, though framed as short-term.

### junior_vs_senior_frontend

_Junior CV (1 year) against a senior / lead frontend offer._

- graph run: 402.2s, review iterations 2, reviewer approved: False
- matched skills: React, TypeScript, Jest, JavaScript
- gaps: Next.js, Cypress, GraphQL, E-commerce
- letter language: fr
- **score_in_range** 1.00 — score 48, expected [30, 65]
- **skills_recall** 1.00 — all found
- **gaps_recall** 1.00 — all flagged
- **letter_faithfulness** 0.90 — The cover letter accurately reflects the CV: it identifies Lucas Ferreira as a junior frontend developer based in Bordeaux, correctly cites Agence Pixelbleu, Figma-to-React integration for showcase sites, initial TypeScript work on an appointment project, and Jest unit testing on main components. It also correctly states no prior management or architecture responsibilities. The only unsupported detail is the target employer 'Vinéo,' which is not in the CV, but it is a job-specific reference rather than a claim about the candidate's background. No invented skills, employers, durations, certifications, or inflated seniority are present.
- **letter_relevance** 0.40 — The letter is in French, professional, well-structured, and concise, and it references specific job elements such as React, TypeScript, JavaScript, Jest, architecture, team support, and testing. However, it does not strongly align with the key requirements because it explicitly states the candidate is junior and lacks lead experience, architecture responsibility, Next.js production experience, and Cypress, rather than linking those core requirements to relevant experience.
- **gap_usefulness** 0.90 — The advice is concrete and tailored for each gap: it uses the candidate's React, TypeScript, Jest, and showcase/appointment project experience, and proposes specific actions such as the official Next.js Learn path, 3-5 Cypress E2E tests, a GraphQL mini-project with Hasura, and a mini e-commerce demo emphasizing accessibility and Core Web Vitals. Minor deduction because the Next.js and Cypress guidance could be more specific to production-level requirements, such as SSR/data fetching or a fuller E2E test strategy.

### overqualified_support

_Over-qualified — an IT manager with 15 years of experience applying to a level-1 helpdesk job._

- graph run: 330.7s, review iterations 2, reviewer approved: False
- matched skills: Windows, Office 365, Active Directory, ITIL, GLPI
- gaps: (none)
- letter language: fr
- **score_in_range** 1.00 — score 96, expected [65, 100]
- **skills_recall** 1.00 — all found
- **gaps_recall** 1.00 — all flagged
- **letter_faithfulness** 1.00 — All factual claims are supported by the CV: the RSSI role, N1 support at Bretagne Informatique Services, AD/GPO/Windows Server administration at Ouest Logistique, the 600-post Windows 10/Office 365 migration, ITIL v4 service center with GLPI, and management of an 8-person team are all directly stated or reasonably implied. No invented employers, skills, figures, certifications, or inflated durations are present.
- **letter_relevance** 1.00 — The letter is in French, professionally structured with greeting, body, and closing, and explicitly links the candidate's experience to the key job requirements: N1 support, GLPI ticketing, Windows/Office 365 incident resolution, Active Directory account management, workstation installation, and ITIL practices. It is concise and tailored to Breizh Mutuelle Services rather than generic.
- **gap_usefulness** 1.00 — The identified gaps section explicitly states none, and the actual output correctly reflects that there are no gaps to address. Per the evaluation steps, this warrants the maximum score.

### english_offer_ml

_Offer written in English (ML engineer, Amsterdam); CV in French. Letter must be in English._

- graph run: 312.2s, review iterations 2, reviewer approved: False
- matched skills: Python, PyTorch, Docker, SQL, Time-series forecasting
- gaps: TensorFlow, Kubernetes, MLflow, Dutch
- letter language: en
- **score_in_range** 1.00 — score 58, expected [50, 85]
- **skills_recall** 1.00 — all found
- **gaps_recall** 1.00 — all flagged
- **letter_faithfulness** 0.90 — The cover letter accurately reflects the CV: Data Scientist role, École Centrale de Lille data science engineering degree, Nordpanier store-level forecasting with time-series, LightGBM, and PyTorch, 12% stockout reduction in fresh perimeter, FastAPI/Docker model exposure, SQL/BigQuery usage, Python/PyTorch/Docker/SQL experience, and C1 English. Minor phrasing such as 'data engineering skills' and 'prepare data for training and evaluation' is reasonably implied by the CV’s SQL, BigQuery, API, and Docker work, with no clear invented employer, figure, certification, or inflated duration.
- **letter_relevance** 0.70 — The letter is in English, professionally structured, and concise. It explicitly connects relevant experience to key job requirements, including time-series forecasting, PyTorch, Docker, SQL, FastAPI serving, and translating business questions into ML solutions. However, it does not explicitly address 3+ years of experience, recommendation models, production monitoring, or MLflow/experiment tracking, and it frames TensorFlow, Kubernetes, and MLflow as skills to learn rather than demonstrated experience.
- **gap_usefulness** 1.00 — Each identified gap receives concrete, actionable advice tied to the candidate's CV: TensorFlow is addressed with a specific course/certification and a PyTorch-to-TensorFlow forecasting project; Kubernetes is addressed by deploying the existing Dockerized FastAPI forecasting API to k3d/Minikube; MLflow is addressed by integrating it into the existing Nordpanier forecasting workflow with a model registry; Dutch is addressed with a realistic A1 course plan and honest application framing. The advice avoids generic recommendations and is proportionate to required and nice-to-have gaps.

### employment_gap_accountant

_CV with a 2-year employment gap (parental leave) — accountant for an accountant offer._

- graph run: 398.0s, review iterations 2, reviewer approved: False
- matched skills: Comptabilité générale, TVA, Bilans, Sage, Excel
- gaps: Pennylane, Paie
- letter language: fr
- **score_in_range** 1.00 — score 71, expected [60, 95]
- **skills_recall** 1.00 — all found
- **gaps_recall** 1.00 — all flagged
- **letter_faithfulness** 0.90 — The cover letter accurately reflects the CV: DCG, cabinet accountant role, 70 months at Cabinet Lemoine Expertise, 35 dossiers, TVA/bilan tasks, and daily Sage/Excel use are all supported. The only weak point is the negative claim that Pennylane is not among daily tools, which is not explicitly stated in the CV, but it is a minor unsupported inference rather than a clear hallucination.
- **letter_relevance** 0.90 — The letter is in French, professionally structured, and directly links the candidate’s cabinet experience to the key missions: general accounting, TVA declarations, balance sheet preparation, Sage/Excel with pivot tables, and client relations. It also transparently addresses Pennylane as a gap with a training intention. Minor shortcomings are the lack of explicit mention of balance sheet revision and the exact 40-client portfolio size, but overall alignment is strong.
- **gap_usefulness** 0.90 — The response addresses both identified gaps with actionable advice. For Pennylane, it suggests a specific resource (Pennylane Academy/official tutorials), a concrete mini-project, and a precise way to leverage the candidate's Sage/Excel and multi-dossier experience. For Paie, it recommends a short, realistic training path and uses the candidate's DCG and accounting background to frame the gap in an interview. The main shortcoming is that the Paie advice is somewhat less specific, as it does not name a particular certification or payroll software.

### few_skills_logistics

_CV with very few explicit skills (no skills section) — warehouse team leader offer._

- graph run: 407.5s, review iterations 2, reviewer approved: False
- matched skills: Logistique, Encadrement, CACES 1, CACES 3, CACES 5
- gaps: WMS, SAP, Excel
- letter language: fr
- **score_in_range** 1.00 — score 61, expected [40, 80]
- **skills_recall** 1.00 — all found
- **gaps_recall** 1.00 — all flagged
- **letter_faithfulness** 0.80 — The cover letter accurately reflects the CV for the main factual claims: current team lead role at Entrepôt Val de Loire Distribution with 8 people, planning, training, rolling inventories, weekly productivity reporting; prior Logicentre roles as cariste and order preparer with truck reception, scanner preparation, forklift driving; CACES 1, 3, 5 with 2023 recycling; and SST training. The only notable unsupported detail is the explicit statement of no direct WMS and SAP experience, which is not stated in the CV, though it is a negative disclaimer rather than an inflated or invented positive claim.
- **letter_relevance** 0.80 — The letter is in French, professionally structured, and concise. It explicitly links the candidate's experience to key missions such as team leadership, inventories, safety, productivity tracking, truck reception, and CACES 1, 3, and 5. However, it does not demonstrate WMS/SAP experience, only acknowledges a gap, and omits error-rate tracking and Excel, reducing full alignment.
- **gap_usefulness** 0.90 — The advice is concrete and tied to the candidate's CV: it suggests naming the WMS used for scanner-based preparation and rolling inventories, adding specific SAP modules/transactions with an openSAP exercise, and leveraging productivity reporting with an Excel dashboard. Minor generic training suggestions prevent a perfect score.
