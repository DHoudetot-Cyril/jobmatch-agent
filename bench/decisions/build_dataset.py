"""Build the labelled decision benchmark (bench/decisions/dataset/*.jsonl).

All texts are FICTIONAL. Sources: evals/dataset/*.yaml (offers + CVs), data/samples/,
letters produced by the graph in evals/results/outputs/*.json, plus short texts written here.

Each item also carries `question_en`: the same question (and option descriptions) in English,
the state left untouched — used by the `<backend>@q-en` ablation of run_bench.py (decision
models are often English-centric; in the graph we control the question language, not the
document language). Skill names / claim quotes stay as written.

Labels are hand-written below (not produced by any model). Every non-obvious item carries a
`note` explaining the label. Splits are stratified per (task, label): ~30% "calib", ~70% "test",
assigned deterministically from a hash of the item id.

    uv run python bench/decisions/build_dataset.py
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "dataset"


def _case(stem: str) -> dict:
    return yaml.safe_load((ROOT / "evals" / "dataset" / f"{stem}.yaml").read_text())


CASES = {
    "backend": _case("01_strong_fit_backend"),
    "devops": _case("02_partial_fit_devops"),
    "nurse": _case("03_mismatch_nurse_data"),
    "frontend": _case("04_junior_vs_senior_frontend"),
    "support": _case("05_overqualified_support"),
    "ml": _case("06_english_offer_ml"),
    "accountant": _case("07_employment_gap_accountant"),
    "logistics": _case("08_few_skills_logistics"),
}

# --------------------------------------------------------------------------------------------
# CVs written for this benchmark (fictional people and companies)
# --------------------------------------------------------------------------------------------
CV_BANK_FR = """Julien Morel (personne fictive)
Analyste opérations bancaires — Nancy — julien.morel@example.com

EXPÉRIENCES
Analyste back-office — Banque de l'Est Lorrain (établissement fictif), Nancy
Mai 2019 – aujourd'hui (5 ans 6 mois)
- Traitement et contrôle des virements internationaux SWIFT (MT103, MT202).
- Contrôles KYC et signalements dans le cadre de la lutte contre le blanchiment (LCB-FT).
- Automatisation des rapprochements de comptes avec des macros VBA sous Excel.
- Requêtes SQL sur la base Oracle de reporting réglementaire.

Conseiller clientèle — Crédit des Vosges (établissement fictif), Épinal
2016 – 2019
- Gestion d'un portefeuille de 400 clients particuliers.

COMPÉTENCES
Excel (VBA), SQL, Oracle, réglementation bancaire, notions de Power BI

FORMATION
Licence Banque-Finance-Assurance — Université de Lorraine (2016)
MOOC « Python pour débutants » (en cours, 2024)

CENTRES D'INTÉRÊT
Joueur au club de go de Nancy (3e kyu), club de lecture (Kafka, Camus, Duras).
"""

CV_LEAD_FR = """Inès Caron (personne fictive)
Tech Lead Java — Lille — ines.caron@example.com

EXPÉRIENCES
Tech Lead — Assurances Flandre Mutuelle (entreprise fictive), Lille
2019 – aujourd'hui (5 ans)
- Encadrement d'une équipe de 6 développeurs, entretiens annuels et recrutements.
- Architecture de microservices Java 17 / Spring Boot pour la gestion des sinistres.
- Échanges asynchrones entre services via Apache Kafka (2 millions de messages par jour).
- Déploiement des services sur le cloud privé OpenShift de l'entreprise.
- Animation des rituels Scrum et du découpage des user stories avec le Product Owner.

Développeuse Java senior — Nordinfo Services (ESN fictive), Lille
2012 – 2019
- Applications Java EE pour des clients du secteur bancaire, base de données Oracle.

COMPÉTENCES
Java, Spring Boot, Kafka, Oracle, OpenShift, Maven, Git, Scrum, management d'équipe

FORMATION
Diplôme d'ingénieur informatique — Polytech Lille (2012)

LANGUES
Français, anglais (professionnel)
"""

CV_IOS_EN = """Emily Carter (fictional person)
Senior iOS Developer — Manchester, UK — emily.carter@example.com

EXPERIENCE
Senior iOS Developer — Paddlefin (fictional fintech), Manchester
March 2019 – present (5 years 8 months)
- Built the budgeting features of the iOS app in Swift and SwiftUI (1.2M monthly users).
- Maintained legacy Objective-C payment modules and migrated two of them to Swift.
- Push notifications and remote config with Firebase.
- CI/CD with Fastlane and Bitrise; unit and UI tests with XCTest.
- Mentored two junior developers.

iOS Developer — Brightlane Studio (fictional agency), Leeds
2016 – 2019
- Client apps in Objective-C and Swift for retail brands.

SKILLS
Swift, SwiftUI, Objective-C, UIKit, Firebase, Fastlane, Bitrise, XCTest, Git

EDUCATION
BSc Computer Science — University of Leeds (2016)
Weekend introduction course to Kotlin (2023)
"""

CV_QA_EN = """Daniel Okafor (fictional person)
Junior QA Engineer — Bristol, UK — daniel.okafor@example.com

EXPERIENCE
QA Engineer — Cartwheel Retail (fictional company), Bristol
September 2023 – present (1 year 2 months)
- Wrote automated UI tests with Selenium WebDriver in Java for the checkout flow.
- Tested REST APIs with Postman collections.
- Tests run nightly in the Jenkins CI pipeline; I triage the failures every morning.
- Manual exploratory testing and bug reports in Jira.

QA Intern — Harbourline Insurance (fictional company), Bristol
April 2023 – August 2023
- Manual regression testing of the customer portal.

SKILLS
Selenium, Java, Postman, Jenkins, Jira, ISTQB Foundation, basic knowledge of SQL

EDUCATION
BSc Mathematics — University of the West of England (2023)
Currently following an online Python course.

INTERESTS
Chess and the board game Go, five-a-side football.
"""

NEW_CVS = {
    "bank": ("fr", CV_BANK_FR),
    "lead": ("fr", CV_LEAD_FR),
    "ios": ("en", CV_IOS_EN),
    "qa": ("en", CV_QA_EN),
}


def cv_text(key: str) -> str:
    if key == "data_engineer":
        return (ROOT / "data" / "samples" / "cv_data_engineer.txt").read_text()
    if key in NEW_CVS:
        return NEW_CVS[key][1]
    return CASES[key]["cv_text"]


def offer_text(key: str) -> str:
    if key == "data_engineer":
        return (ROOT / "data" / "samples" / "offer_data_engineer.txt").read_text()
    return CASES[key]["offer_text"]


# --------------------------------------------------------------------------------------------
# skill_evidence  (yes_no)
# --------------------------------------------------------------------------------------------
SKILL_Q = {
    "fr": (
        "Le CV montre-t-il que la personne a réellement pratiqué « {s} » ? "
        "(Compte comme oui : une utilisation dans un poste, un stage, une alternance ou un projet "
        "concret, ou une compétence listée sans réserve. Ne compte pas : de simples « notions », "
        "un cours suivi ou en cours, un loisir, ou un homonyme.)"
    ),
    "en": (
        'Does the CV show that the person has actually used "{s}"? '
        "(Counts as yes: use in a job, internship or concrete project, or a skill listed without "
        "qualification. Does not count: mere 'basic knowledge', a course taken or in progress, "
        "a hobby, or a homonym.)"
    ),
}

# (cv_key, skill, label, note)
SKILLS: list[tuple[str, str, str, str]] = [
    # Hugo — backend
    ("backend", "Django", "yes", ""),
    ("backend", "PostgreSQL", "yes", ""),
    ("backend", "MongoDB", "no", "Absent; close to PostgreSQL (database) which is present."),
    ("backend", "FastAPI", "no", "Absent; Flask and Django present (close Python web frameworks)."),
    ("backend", "Kubernetes", "no", "Absent; only Docker images. Offer lists it as a gap."),
    ("backend", "SQL", "yes", "Synonym-level: optimising PostgreSQL queries implies SQL."),
    ("backend", "intégration continue (CI/CD)", "yes", "Paraphrase: GitHub Actions pipelines."),
    ("backend", "conteneurisation", "yes", "Paraphrase: builds Docker images."),
    ("backend", "files de tâches asynchrones", "yes", "Paraphrase: Celery + Redis tasks."),
    ("backend", "Java", "no", "Absent."),
    # Nadia — sysadmin / devops
    ("devops", "Ansible", "yes", ""),
    ("devops", "Docker", "yes", "Docker Compose services in production + listed."),
    ("devops", "Kubernetes", "no", "Absent; only Docker Compose (close container tech)."),
    ("devops", "Terraform", "no", "Absent; Ansible is a close IaC/config tool."),
    ("devops", "Python", "no", "Only 'Python (notions)' in the skills list, no concrete use."),
    ("devops", "supervision / monitoring", "yes", "Paraphrase: Zabbix supervision."),
    ("devops", "gestion de configuration automatisée", "yes", "Paraphrase: 60 Ansible roles."),
    ("devops", "virtualisation", "yes", "VMware vSphere."),
    ("devops", "AWS", "no", "Absent; servers are on-premise ('sur site')."),
    # Élodie — nurse
    ("nurse", "Excel", "yes", "Listed under Pack Office (Word, Excel) without qualification."),
    ("nurse", "Python", "no", "Absent (only appears in the offer, not the CV)."),
    ("nurse", "formation / pédagogie", "yes", "Trains students and colleagues (new software)."),
    ("nurse", "SQL", "no", "Absent."),
    ("nurse", "gestion de l'urgence", "yes", "Listed + ICU work."),
    # Lucas — junior frontend
    ("frontend", "React", "yes", ""),
    ("frontend", "TypeScript", "yes", "Listed '(débutant)' but concrete use on a project -> yes."),
    ("frontend", "Vue.js", "no", "Absent; React is a close framework."),
    ("frontend", "Next.js", "no", "Absent; React (related) is present."),
    ("frontend", "tests unitaires", "yes", "Jest unit tests."),
    ("frontend", "Angular", "no", "Absent."),
    # Patrick — IT manager
    ("support", "ITIL", "yes", "ITIL v4 service centre + Foundation certification."),
    ("support", "Active Directory", "yes", ""),
    ("support", "Linux", "no", "Absent; only Windows / Windows Server."),
    ("support", "Azure", "no", "Absent; Office 365 is Microsoft cloud but not Azure."),
    ("support", "gestion budgétaire", "yes", "Annual budget of 1.2 M€."),
    ("support", "management d'équipe", "yes", "Led a team of 8."),
    # Sofiane — data scientist (FR CV)
    ("ml", "PyTorch", "yes", ""),
    ("ml", "TensorFlow", "no", "Absent; PyTorch is the close alternative present."),
    ("ml", "Kubernetes", "no", "Absent; models served with Docker only."),
    ("ml", "machine learning", "yes", "Paraphrase: forecasting models with LightGBM/PyTorch."),
    ("ml", "Google Cloud", "yes", "Hard: BigQuery is a Google Cloud product used in the job."),
    ("ml", "MLflow", "no", "Absent."),
    ("ml", "séries temporelles", "yes", ""),
    ("ml", "FastAPI", "yes", ""),
    ("ml", "Spark", "no", "Absent (pandas, BigQuery only)."),
    # Claire — accountant
    ("accountant", "Sage", "yes", ""),
    ("accountant", "SAP", "no", "Absent; Sage is the close accounting software present."),
    ("accountant", "paie", "no", "Absent: payroll never mentioned."),
    ("accountant", "TVA", "yes", ""),
    ("accountant", "tableaux croisés dynamiques", "yes", "Excel pivot tables explicitly used."),
    # Mickaël — logistics
    ("logistics", "CACES", "yes", "CACES 1, 3, 5."),
    ("logistics", "management d'équipe", "yes", "Chef d'équipe, team of 8."),
    ("logistics", "Excel", "no", "Never mentioned (reports figures, tool unspecified)."),
    ("logistics", "SAP", "no", "Absent; only 'préparation sur scanner'."),
    ("logistics", "gestion des stocks / inventaires", "yes", "Inventaires tournants."),
    # Camille — data engineer
    ("data_engineer", "Airflow", "yes", ""),
    ("data_engineer", "Spark", "yes", "PySpark pipelines."),
    ("data_engineer", "Kafka", "no", "Absent; streaming not mentioned in the CV."),
    ("data_engineer", "Azure", "no", "Absent; AWS is the close cloud present."),
    ("data_engineer", "MySQL", "no", "Absent; PostgreSQL is the close DB present."),
    ("data_engineer", "dbt", "no", "Absent."),
    ("data_engineer", "data visualisation", "yes", "Power BI dashboards for clients."),
    ("data_engineer", "conteneurisation", "yes", "Jobs containerised with Docker."),
    ("data_engineer", "qualité des données", "yes", "Great Expectations tests."),
    ("data_engineer", "Kubernetes", "no", "Absent; Docker + GitLab CI only."),
    # Julien — bank (homonyms & hobbies)
    ("bank", "Swift", "no", "Homonym: SWIFT bank transfers, not the Swift programming language."),
    ("bank", "Go", "no", "Homonym: plays the board game go, not the Go language."),
    ("bank", "Apache Kafka", "no", "Homonym: reads Kafka (the writer) in a book club."),
    ("bank", "Python", "no", "Only a beginner MOOC still in progress."),
    ("bank", "Power BI", "no", "Only 'notions de Power BI'."),
    ("bank", "VBA", "yes", "Excel VBA macros for reconciliations."),
    ("bank", "SQL", "yes", ""),
    ("bank", "PostgreSQL", "no", "Absent; Oracle is the close DB present."),
    ("bank", "lutte anti-blanchiment", "yes", "LCB-FT / KYC controls."),
    ("bank", "Oracle", "yes", ""),
    # Inès — lead
    ("lead", "Spring Boot", "yes", ""),
    ("lead", "Kafka", "yes", "Apache Kafka between microservices."),
    (
        "lead",
        "Kubernetes",
        "yes",
        "Hard synonym: deploys on OpenShift, a Kubernetes distribution.",
    ),
    ("lead", "PostgreSQL", "no", "Absent; Oracle is the close DB present."),
    ("lead", "AWS", "no", "Absent; private OpenShift cloud."),
    ("lead", "Scrum", "yes", ""),
    ("lead", "management d'équipe", "yes", "Manages 6 developers."),
    ("lead", "Python", "no", "Absent."),
    ("lead", "architecture microservices", "yes", ""),
    # Emily — iOS (EN)
    ("ios", "Swift", "yes", ""),
    ("ios", "SwiftUI", "yes", ""),
    ("ios", "Objective-C", "yes", "Legacy modules maintained."),
    ("ios", "Kotlin", "no", "Only a weekend introduction course."),
    ("ios", "Android development", "no", "Absent; iOS only."),
    ("ios", "React Native", "no", "Absent."),
    ("ios", "Firebase", "yes", ""),
    ("ios", "CI/CD", "yes", "Fastlane + Bitrise."),
    ("ios", "mentoring", "yes", "Mentored two juniors."),
    # Daniel — QA (EN)
    ("qa", "Selenium", "yes", ""),
    ("qa", "Java", "yes", "Selenium tests written in Java."),
    ("qa", "API testing", "yes", "REST APIs tested with Postman."),
    ("qa", "Cypress", "no", "Absent; Selenium is the close tool present."),
    ("qa", "Go", "no", "Homonym: board game Go in interests."),
    ("qa", "Python", "no", "Online course in progress only."),
    ("qa", "SQL", "no", "Only 'basic knowledge of SQL'."),
    ("qa", "Jenkins", "yes", "Triages nightly Jenkins runs."),
    ("qa", "performance testing", "no", "Absent (no JMeter/load testing)."),
]


def lang_of(cv_key: str) -> str:
    return NEW_CVS[cv_key][0] if cv_key in NEW_CVS else "fr"


def build_skill_evidence() -> list[dict]:
    items = []
    for i, (cv, skill, label, note) in enumerate(SKILLS):
        lang = lang_of(cv)
        items.append(
            {
                "id": f"skill-{i:03d}-{cv}",
                "task": "skill_evidence",
                "lang": lang,
                "state": cv_text(cv),
                "question": {"kind": "yes_no", "question": SKILL_Q[lang].format(s=skill)},
                "question_en": {"kind": "yes_no", "question": SKILL_Q["en"].format(s=skill)},
                "label": label,
                "source": f"cv:{cv}",
                "note": note,
            }
        )
    return items


# --------------------------------------------------------------------------------------------
# claim_supported  (yes_no)
# --------------------------------------------------------------------------------------------
CLAIM_Q = {
    "fr": (
        "Voici une phrase extraite de la lettre de motivation de cette personne : « {c} »\n"
        "Cette affirmation est-elle entièrement étayée par le CV (aucun fait, chiffre, durée, "
        "rôle ou collaboration inventé ou exagéré) ?"
    ),
    "en": (
        'Here is a sentence from this person\'s cover letter: "{c}"\n'
        "Is this claim fully supported by the CV (no invented or exaggerated fact, number, "
        "duration, role or collaboration)?"
    ),
}

# (cv_key, lang, claim, label, source, note)
CLAIMS: list[tuple[str, str, str, str, str, str]] = [
    # Hugo — from the generated letter (strong_fit_backend.json) + crafted
    (
        "backend",
        "fr",
        "Chez Parcelia, j'ai développé des API REST avec Django REST Framework pour "
        "40 000 commandes par jour.",
        "yes",
        "letter",
        "",
    ),
    (
        "backend",
        "fr",
        "J'ai optimisé des requêtes PostgreSQL, ce qui a divisé par trois le temps "
        "de réponse moyen.",
        "yes",
        "letter",
        "",
    ),
    (
        "backend",
        "fr",
        "J'ai réalisé des tests unitaires et d'intégration avec pytest, portant la "
        "couverture de 45 % à 82 %.",
        "yes",
        "letter",
        "",
    ),
    (
        "backend",
        "fr",
        "Chez Parcelia, j'ai développé des API REST pour 400 000 commandes par jour.",
        "no",
        "crafted",
        "Wrong number: 40 000 in the CV.",
    ),
    (
        "backend",
        "fr",
        "J'ai porté la couverture de tests de 45 % à 92 %.",
        "no",
        "crafted",
        "Wrong number: 82 % in the CV.",
    ),
    (
        "backend",
        "fr",
        "J'ai encadré une équipe de quatre développeurs chez Parcelia.",
        "no",
        "crafted",
        "Invented management role.",
    ),
    (
        "backend",
        "fr",
        "Fort de six ans d'expérience en développement Django, je maîtrise les API à fort volume.",
        "no",
        "crafted",
        "Inflated: ~3y10m Django at Parcelia (+16 months Flask/scripts in alternance).",
    ),
    (
        "backend",
        "fr",
        "J'ai déployé nos API sur un cluster Kubernetes en production.",
        "no",
        "crafted",
        "Kubernetes absent from CV.",
    ),
    (
        "backend",
        "fr",
        "J'ai travaillé en étroite collaboration avec l'équipe data science de "
        "Parcelia sur la détection de fraude.",
        "no",
        "crafted",
        "Invented collaboration.",
    ),
    (
        "backend",
        "fr",
        "J'ai mis en place des tâches asynchrones avec Celery et Redis pour l'envoi "
        "d'e-mails et la génération de factures.",
        "yes",
        "letter",
        "",
    ),
    # Sofiane — English letter (english_offer_ml.json) + crafted
    (
        "ml",
        "en",
        "At Nordpanier, I developed store-level sales forecasting models using "
        "time-series methods, LightGBM, and PyTorch.",
        "yes",
        "letter",
        "",
    ),
    (
        "ml",
        "en",
        "This work contributed to a 12% reduction in stockouts for the fresh product perimeter.",
        "yes",
        "letter",
        "",
    ),
    (
        "ml",
        "en",
        "I exposed the models through a FastAPI service containerized with Docker.",
        "yes",
        "letter",
        "",
    ),
    (
        "ml",
        "en",
        "This work reduced stockouts by 20% across all stores.",
        "no",
        "crafted",
        "Wrong number and scope: 12 % on fresh products.",
    ),
    (
        "ml",
        "en",
        "I led a team of five data scientists at Nordpanier.",
        "no",
        "crafted",
        "Invented leadership.",
    ),
    (
        "ml",
        "en",
        "I deployed and monitored our models on Kubernetes with MLflow tracking.",
        "no",
        "crafted",
        "Kubernetes and MLflow absent.",
    ),
    (
        "ml",
        "en",
        "With five years of experience as a data scientist, I can own production ML end to end.",
        "no",
        "crafted",
        "Inflated: 3y5m as Data Scientist (+ alternance as Data Analyst).",
    ),
    ("ml", "en", "I work comfortably in English at a C1 level.", "yes", "letter", ""),
    # Élodie — nurse letter (mismatch_nurse_data.json) + crafted
    (
        "nurse",
        "fr",
        "En service de réanimation, j'ai pris en charge des patients en soins "
        "critiques sur 12 lits.",
        "yes",
        "letter",
        "",
    ),
    ("nurse", "fr", "J'ai également encadré des étudiants infirmiers.", "yes", "letter", ""),
    (
        "nurse",
        "fr",
        "J'ai participé à la mise en place d'un logiciel de dossier patient "
        "informatisé et à la formation de mes collègues.",
        "yes",
        "letter",
        "",
    ),
    (
        "nurse",
        "fr",
        "J'ai piloté le déploiement du logiciel de dossier patient informatisé dans "
        "tout l'hôpital.",
        "no",
        "crafted",
        "Inflated: she 'participated', did not lead it.",
    ),
    (
        "nurse",
        "fr",
        "Forte de dix ans d'expérience en réanimation, je connais parfaitement les "
        "soins critiques.",
        "no",
        "crafted",
        "Inflated: ICU since 2017 (~7 years); 2014-2017 was general medicine.",
    ),
    (
        "nurse",
        "fr",
        "J'ai déjà analysé des données hospitalières avec Python et SQL.",
        "no",
        "crafted",
        "Python/SQL absent from the CV.",
    ),
    # Camille — data engineer (crafted, letter-style)
    (
        "data_engineer",
        "fr",
        "Chez Transfluvia, j'ai migré les traitements cron vers Apache Airflow (45 DAGs).",
        "yes",
        "crafted",
        "",
    ),
    (
        "data_engineer",
        "fr",
        "Mes pipelines PySpark traitent 15 millions d'événements de suivi de colis par jour.",
        "yes",
        "crafted",
        "",
    ),
    (
        "data_engineer",
        "fr",
        "Mes pipelines PySpark traitent 50 millions d'événements par jour.",
        "no",
        "crafted",
        "Wrong number: 15 millions.",
    ),
    (
        "data_engineer",
        "fr",
        "J'ai mis en place des pipelines de streaming temps réel avec Kafka.",
        "no",
        "crafted",
        "Kafka/streaming absent.",
    ),
    (
        "data_engineer",
        "fr",
        "J'ai industrialisé les modèles de prévision en collaboration avec "
        "les data scientists de Transfluvia.",
        "no",
        "crafted",
        "Invented collaboration (it's a mission of the OFFER, not in the CV).",
    ),
    (
        "data_engineer",
        "fr",
        "J'ai créé des tableaux de bord Power BI pour des clients du secteur public.",
        "yes",
        "crafted",
        "",
    ),
    (
        "data_engineer",
        "fr",
        "Forte de quatre ans d'expérience en data engineering, je conçois "
        "des pipelines pour des environnements à fort volume.",
        "yes",
        "crafted",
        "CV profile states 4 years of experience (2020-2024, incl. BI role and internship).",
    ),
    # Nadia — devops
    (
        "devops",
        "fr",
        "J'administre un parc de 120 serveurs Linux et j'ai écrit 60 rôles Ansible.",
        "yes",
        "crafted",
        "",
    ),
    (
        "devops",
        "fr",
        "J'ai conteneurisé l'ensemble des applications de la clinique et les "
        "orchestre avec Kubernetes.",
        "no",
        "crafted",
        "Inflated: 'quelques services' with Docker Compose; no Kubernetes.",
    ),
    (
        "devops",
        "fr",
        "Je développe régulièrement des outils internes en Python.",
        "no",
        "crafted",
        "Only 'Python (notions)'; tooling is in Bash.",
    ),
    (
        "devops",
        "fr",
        "J'ai mis en place la supervision avec Zabbix.",
        "yes",
        "crafted",
        "CV: 'Supervision avec Zabbix'. Slight wording stretch but supported.",
    ),
    # Lucas — junior frontend
    ("frontend", "fr", "J'intègre des maquettes Figma en composants React.", "yes", "crafted", ""),
    (
        "frontend",
        "fr",
        "Avec trois ans d'expérience en React, je suis autonome sur des "
        "architectures frontend complexes.",
        "no",
        "crafted",
        "Inflated: ~1 year + 4-month intern.",
    ),
    (
        "frontend",
        "fr",
        "J'ai écrit des tests unitaires avec Jest sur les composants principaux.",
        "yes",
        "crafted",
        "",
    ),
    # Patrick — IT manager
    (
        "support",
        "fr",
        "J'ai dirigé une équipe de 8 personnes et piloté la migration de 600 "
        "postes vers Windows 10.",
        "yes",
        "crafted",
        "",
    ),
    (
        "support",
        "fr",
        "J'ai géré un budget annuel de 2 M€.",
        "no",
        "crafted",
        "Wrong number: 1,2 M€.",
    ),
    ("support", "fr", "Je suis certifié ITIL v4 Foundation.", "yes", "crafted", ""),
    # Claire — accountant
    (
        "accountant",
        "fr",
        "J'ai tenu la comptabilité générale de 35 dossiers en cabinet.",
        "yes",
        "crafted",
        "",
    ),
    (
        "accountant",
        "fr",
        "J'ai tenu la comptabilité de 35 dossiers pendant huit ans en cabinet.",
        "no",
        "crafted",
        "Inflated duration: 5y10m in the cabinet.",
    ),
    # Mickaël — logistics
    (
        "logistics",
        "fr",
        "J'organise le travail d'une équipe de 8 personnes et je forme les nouveaux arrivants.",
        "yes",
        "crafted",
        "",
    ),
    (
        "logistics",
        "fr",
        "J'ai réduit de 30 % les erreurs de préparation de l'entrepôt.",
        "no",
        "crafted",
        "Invented achievement / number.",
    ),
    # Inès — lead
    ("lead", "fr", "J'encadre une équipe de 6 développeurs Java.", "yes", "crafted", ""),
    (
        "lead",
        "fr",
        "J'ai conçu une architecture microservices traitant 20 millions de messages "
        "Kafka par jour.",
        "no",
        "crafted",
        "Wrong number: 2 millions.",
    ),
    # Emily — iOS (EN)
    (
        "ios",
        "en",
        "I built the budgeting features of an iOS app used by 1.2 million monthly users.",
        "yes",
        "crafted",
        "",
    ),
    (
        "ios",
        "en",
        "I managed a team of six iOS developers at Paddlefin.",
        "no",
        "crafted",
        "Inflated: mentored two juniors, no management.",
    ),
    (
        "ios",
        "en",
        "I migrated two legacy Objective-C payment modules to Swift.",
        "yes",
        "crafted",
        "",
    ),
    # Daniel — QA (EN)
    (
        "qa",
        "en",
        "I write automated UI tests with Selenium WebDriver in Java.",
        "yes",
        "crafted",
        "",
    ),
    (
        "qa",
        "en",
        "I designed the company's whole CI pipeline in Jenkins.",
        "no",
        "crafted",
        "Inflated: he triages nightly runs, did not design the pipeline.",
    ),
]


def build_claims() -> list[dict]:
    items = []
    for i, (cv, lang, claim, label, src, note) in enumerate(CLAIMS):
        items.append(
            {
                "id": f"claim-{i:03d}-{cv}",
                "task": "claim_supported",
                "lang": lang,
                "state": cv_text(cv),
                "question": {"kind": "yes_no", "question": CLAIM_Q[lang].format(c=claim)},
                "question_en": {"kind": "yes_no", "question": CLAIM_Q["en"].format(c=claim)},
                "label": label,
                "source": f"cv:{cv};{src}",
                "note": note,
            }
        )
    return items


# --------------------------------------------------------------------------------------------
# seniority  (choice)
# --------------------------------------------------------------------------------------------
SEN_Q = {
    "fr": "Quel est le niveau de séniorité de cette personne d'après son CV ?",
    "en": "What is this person's seniority level according to the CV?",
}
SEN_OPTS = {
    "fr": {
        "junior": "Junior : moins de 3 ans d'expérience professionnelle dans le métier",
        "mid": "Confirmé : 3 à 5 ans d'expérience, autonome",
        "senior": "Senior : 6 ans ou plus d'expérience, expert, sans encadrement hiérarchique",
        "lead": "Lead / manager : encadre une équipe (tech lead, chef d'équipe, responsable)",
    },
    "en": {
        "junior": "Junior: less than 3 years of professional experience in the role",
        "mid": "Mid-level: 3 to 5 years of experience, autonomous",
        "senior": "Senior: 6+ years of experience, expert individual contributor, no reports",
        "lead": "Lead / manager: manages a team (tech lead, team leader, head of)",
    },
}

SNIPPETS = {
    "s_jr_fr": (
        "fr",
        """Manon Petit (personne fictive) — Assistante RH — Dijon
EXPÉRIENCE
Assistante ressources humaines — Bourgogne Textiles (entreprise fictive), depuis septembre 2023
- Suivi des absences, préparation des éléments de paie, organisation des entretiens.
Stage RH — Mairie de Beaune (stage fictif), 6 mois en 2023.
FORMATION
Licence professionnelle Gestion des ressources humaines (2023)""",
    ),
    "s_mid_fr": (
        "fr",
        """Karim Aït-Ali (personne fictive) — Chargé de communication — Marseille
EXPÉRIENCE
Chargé de communication — Office du tourisme des Calanques (fictif), 2020 – aujourd'hui
- Réseaux sociaux, communiqués de presse, organisation de 15 événements par an.
Assistant communication (alternance) — Provence Événements (fictif), 2018 – 2020
FORMATION
Master Communication des organisations — Aix-Marseille Université (2020)""",
    ),
    "s_sr_fr": (
        "fr",
        """Sylvie Marchand (personne fictive) — Juriste droit social — Lyon
EXPÉRIENCE
Juriste droit social — Groupe Rhône Bâtiment (entreprise fictive), 2011 – aujourd'hui
- Conseil aux directions sur les contentieux prud'homaux et les accords collectifs.
- Référente juridique sur les relations avec les instances représentatives du personnel.
Juriste junior — Cabinet Arpège Avocats (fictif), 2008 – 2011
FORMATION
Master 2 Droit social — Université Lyon 3 (2008)""",
    ),
    "s_lead_fr": (
        "fr",
        """Paul Girard (personne fictive) — Responsable d'agence — Tours
EXPÉRIENCE
Responsable d'agence bancaire — Banque du Val (fictif), 2018 – aujourd'hui
- Management de 9 conseillers, objectifs commerciaux, recrutement.
Conseiller clientèle professionnels — Banque du Val, 2012 – 2018
FORMATION
Master Banque — IAE de Tours (2012)""",
    ),
    "s_jr_en": (
        "en",
        """Olivia Bennett (fictional person) — Graduate Data Analyst — Glasgow
EXPERIENCE
Graduate Data Analyst — Thistle Energy (fictional), since October 2024
- SQL queries and Tableau dashboards for the customer team.
Summer intern — Clydeport Logistics (fictional), June – August 2023
EDUCATION
MSc Data Analytics — University of Strathclyde (2024)""",
    ),
    "s_lead_en": (
        "en",
        """Marcus Reid (fictional person) — Engineering Manager — Dublin
EXPERIENCE
Engineering Manager — Larkspur Payments (fictional), 2020 – present
- Manages two squads (11 engineers), hiring, career growth, delivery planning.
Senior Backend Engineer — Larkspur Payments, 2015 – 2020
Backend Engineer — Quayside Software (fictional), 2011 – 2015
EDUCATION
BSc Computer Science — Trinity College Dublin (2011)""",
    ),
}

# (key, label, note) — key is a cv key or a snippet key
SENIORITY: list[tuple[str, str, str]] = [
    ("backend", "mid", "~3y10m as backend dev + 16-month alternance."),
    ("devops", "senior", "5y8m current role + 2.5y technician (~8 years), no reports."),
    ("frontend", "junior", "1 year + internship."),
    ("support", "lead", "Head of IS, managed 8 people."),
    ("ml", "mid", "3y5m Data Scientist + alternance."),
    ("logistics", "lead", "Chef d'équipe, 8 people (since 2021)."),
    ("data_engineer", "mid", "4 years incl. internship."),
    ("bank", "senior", "Hard: 5y6m analyst + 3y adviser (~8.5 y in banking), no reports."),
    ("lead", "lead", "Tech lead managing 6."),
    ("ios", "senior", "Senior iOS, 8+ years, mentoring only (no management)."),
    ("qa", "junior", "1y2m + internship."),
    ("nurse", "senior", "Nurse since 2014 (10 y); supervising students is not team management."),
    ("s_jr_fr", "junior", ""),
    ("s_mid_fr", "mid", "~4.5 years (2020-) + 2y alternance."),
    ("s_sr_fr", "senior", "16 years, referent, no team."),
    ("s_lead_fr", "lead", "Manages 9 advisers."),
    ("s_jr_en", "junior", ""),
    ("s_lead_en", "lead", "Engineering manager of 11."),
    (
        "accountant",
        "senior",
        "Hard: ~8 years accounting (2014-2022) despite a 2-year career "
        "break; no team. Could be argued 'mid' -> spot-check.",
    ),
    ("s_mid_en_x", "mid", "~4 years as product designer + internship."),
]

SNIPPETS["s_mid_en_x"] = (
    "en",
    """Priya Shah (fictional person) — Product Designer — Leicester
EXPERIENCE
Product Designer — Fernway Travel (fictional), January 2021 – present
- User research, Figma prototypes and the design system of the booking app.
Design intern — Copperleaf Studio (fictional), 2020 (6 months)
EDUCATION
BA Graphic Design — De Montfort University (2020)""",
)


def _text_lang(key: str) -> tuple[str, str]:
    if key in SNIPPETS:
        return SNIPPETS[key][1], SNIPPETS[key][0]
    return cv_text(key), lang_of(key)


def build_seniority() -> list[dict]:
    items = []
    for i, (key, label, note) in enumerate(SENIORITY):
        text, lang = _text_lang(key)
        items.append(
            {
                "id": f"sen-{i:03d}-{key}",
                "task": "seniority",
                "lang": lang,
                "state": text,
                "question": {"kind": "choice", "question": SEN_Q[lang], "options": SEN_OPTS[lang]},
                "question_en": {
                    "kind": "choice",
                    "question": SEN_Q["en"],
                    "options": SEN_OPTS["en"],
                },
                "label": label,
                "source": f"cv:{key}",
                "note": note,
            }
        )
    return items


# --------------------------------------------------------------------------------------------
# doc_type  (choice)
# --------------------------------------------------------------------------------------------
DOC_Q = {
    "fr": "De quel type de document s'agit-il ?",
    "en": "What type of document is this?",
}
DOC_OPTS = {
    "fr": {
        "cv": "Un CV / curriculum vitae décrivant le parcours d'une personne",
        "job_offer": "Une offre d'emploi publiée par un employeur",
        "other": "Autre chose (lettre de motivation, e-mail, article, facture, ...)",
    },
    "en": {
        "cv": "A CV / résumé describing a person's background",
        "job_offer": "A job offer / job posting published by an employer",
        "other": "Something else (cover letter, e-mail, article, invoice, ...)",
    },
}

OTHER_DOCS = {
    "o_invoice": (
        "fr",
        "other",
        "",
        """FACTURE N° 2024-0417
Garage Martel (entreprise fictive) — 12 rue des Lilas, 37000 Tours
Client : M. Bernard Lefort (personne fictive)
Vidange + filtre à huile ........ 89,00 € HT
Plaquettes de frein avant ....... 145,00 € HT
Total HT : 234,00 € — TVA 20 % : 46,80 € — Total TTC : 280,80 €
Règlement à réception.""",
    ),
    "o_post": (
        "fr",
        "other",
        "Hard: a person announcing they are job hunting with their skills — not a CV "
        "and not an offer.",
        """Bonjour à toutes et à tous !
Après 4 belles années chez Parcelia (entreprise fictive), je quitte l'aventure et je suis à
l'écoute de nouvelles opportunités en développement backend Python sur Lyon. Django, PostgreSQL,
Docker n'ont plus de secrets pour moi. N'hésitez pas à partager ce post, merci !
#OpenToWork""",
    ),
    "o_news": (
        "fr",
        "other",
        "Hard: article about the job market mentioning recruitment and skills.",
        """Marché de l'emploi tech : les data engineers toujours très recherchés
Selon une étude fictive de l'Observatoire régional du numérique, les offres de data engineer
ont progressé de 18 % en un an dans les Pays de la Loire. Les recruteurs citent Python, SQL et
Airflow comme compétences les plus demandées, tandis que les salaires d'embauche se stabilisent
autour de 45 k€ pour les profils confirmés.""",
    ),
    "o_email": (
        "en",
        "other",
        "Interview scheduling e-mail, recruitment context.",
        """Subject: Interview — Machine Learning Engineer
Hi Sofiane,
Thanks for your application to Tulipwise (fictional company). We'd like to invite you to a
45-minute video interview with our ML lead next Tuesday at 10:00 CET. Please confirm whether
this slot works for you.
Best regards,
Anna de Vries — Talent Acquisition""",
    ),
    "o_menu": (
        "en",
        "other",
        "",
        """THE COPPER KETTLE (fictional restaurant) — LUNCH MENU
Starters: leek and potato soup £6 · smoked mackerel pâté £8
Mains: fish and chips £15 · mushroom risotto £14 · steak pie £16
Desserts: sticky toffee pudding £7 · Eton mess £6
Ask our staff about allergens.""",
    ),
}

# (kind, key) where kind in {"offer","cv","letter","other"}
DOC_ITEMS: list[tuple[str, str]] = [
    ("offer", "backend"),
    ("offer", "devops"),
    ("offer", "nurse"),
    ("offer", "frontend"),
    ("offer", "ml"),
    ("offer", "logistics"),
    ("offer", "data_engineer"),
    ("cv", "support"),
    ("cv", "accountant"),
    ("cv", "data_engineer"),
    ("cv", "lead"),
    ("cv", "qa"),
    ("cv", "s_mid_fr"),
    ("letter", "strong_fit_backend"),
    ("letter", "english_offer_ml"),
    ("other", "o_invoice"),
    ("other", "o_post"),
    ("other", "o_news"),
    ("other", "o_email"),
    ("other", "o_menu"),
]


def _offer_lang(key: str) -> str:
    return "en" if key == "ml" else "fr"


def build_doc_type() -> list[dict]:
    items = []
    for i, (kind, key) in enumerate(DOC_ITEMS):
        note = ""
        if kind == "offer":
            text, lang, label = offer_text(key), _offer_lang(key), "job_offer"
        elif kind == "cv":
            text, lang = _text_lang(key)
            label = "cv"
        elif kind == "letter":
            d = json.loads((ROOT / "evals/results/outputs" / f"{key}.json").read_text())
            text, lang, label = d["cover_letter"]["body"], d["cover_letter"]["language"], "other"
            note = "Hard: cover letter (talks about a candidate's experience) -> other."
        else:
            lang, label, note, text = OTHER_DOCS[key]
        items.append(
            {
                "id": f"doc-{i:03d}-{kind}-{key}",
                "task": "doc_type",
                "lang": lang,
                "state": text,
                "question": {"kind": "choice", "question": DOC_Q[lang], "options": DOC_OPTS[lang]},
                "question_en": {
                    "kind": "choice",
                    "question": DOC_Q["en"],
                    "options": DOC_OPTS["en"],
                },
                "label": label,
                "source": f"{kind}:{key}",
                "note": note,
            }
        )
    return items


# --------------------------------------------------------------------------------------------
def assign_splits(items: list[dict], calib_frac: float = 0.3) -> None:
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for it in items:
        strata[(it["task"], it["label"])].append(it)
    for group in strata.values():
        group.sort(key=lambda it: hashlib.sha256(it["id"].encode()).hexdigest())
        n_cal = round(calib_frac * len(group))
        for j, it in enumerate(group):
            it["split"] = "calib" if j < n_cal else "test"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = {
        "skill_evidence": build_skill_evidence(),
        "claim_supported": build_claims(),
        "seniority": build_seniority(),
        "doc_type": build_doc_type(),
    }
    for name, items in tasks.items():
        assign_splits(items)
        ids = [it["id"] for it in items]
        assert len(ids) == len(set(ids)), name
        with (OUT / f"{name}.jsonl").open("w") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        counts: dict[tuple, int] = defaultdict(int)
        for it in items:
            counts[(it["lang"], it["split"], it["label"])] += 1
        print(name, len(items), dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
