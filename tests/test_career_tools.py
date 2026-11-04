"""Career-advisor tools: local ESCO search, learning-resource lookup, CV evidence (no network)."""

import json
from pathlib import Path

import pytest

from jobmatch.career import tools
from jobmatch.config import get_settings

ROOT = Path(__file__).resolve().parents[1]
CAREER = ROOT / "data" / "career"
CV = "Data engineer. Python, SQL, Apache Spark (PySpark), Apache Airflow, Docker."

ESCO_FIXTURE = {
    "seeds": {"Excel": ["u:sheet"], "Kafka": []},
    "skills": [
        {
            "uri": "u:acc",
            "preferred_label": {"en": "accounting", "fr": "comptabilité"},
            "alt_labels": {"en": ["bookkeeping"], "fr": ["tenue des comptes"]},
            "description": "x" * 1000,
            "related": [
                {"uri": f"u:r{i}", "label": f"rel {i}", "relation": "optional_for"}
                for i in range(20)
            ],
        },
        {
            "uri": "u:py",
            "preferred_label": {"en": "Python (computer programming)", "fr": "Python"},
            "alt_labels": {"en": ["Python3"], "fr": []},
            "description": "Python.",
            "related": [],
        },
        {
            "uri": "u:sheet",
            "preferred_label": {"en": "use spreadsheets software", "fr": "utiliser un tableur"},
            "alt_labels": {"en": [], "fr": []},
            "description": "",
            "related": [],
        },
        {
            "uri": "u:pg",
            "preferred_label": {"en": "PostgreSQL", "fr": "PostgreSQL"},
            "alt_labels": {"en": [], "fr": []},
            "description": "",
            "related": [],
        },
        {
            "uri": "u:excellence",
            "preferred_label": {"en": "demand excellence", "fr": "exiger l'excellence"},
            "alt_labels": {"en": [], "fr": []},
            "description": "",
            "related": [],
        },
    ],
}


def _res(skill, title, aliases=(), free=True, kind="docs"):
    return {
        "skill": skill,
        "aliases": list(aliases),
        "title": title,
        "url": f"https://example.org/{title}",
        "provider": "P",
        "kind": kind,
        "free": free,
        "level": "beginner",
        "language": "en",
        "est_hours": None,
        "verified_at": "2026-11-04",
    }


RES_FIXTURE = {
    "resources": [
        _res("Kafka", "kafka-docs", ["Apache Kafka"]),
        _res("Kafka", "kafka-course", kind="course"),
        _res("Kafka", "kafka-cert", free=False, kind="certification"),
        _res("Cypress", "cypress", ["tests e2e", "end-to-end testing"]),
        _res("Apache Spark", "spark", ["PySpark"]),
        _res("Apache Spark", "spark-quickstart", kind="tutorial"),
        _res("SQL Server", "mssql"),
        _res("SQL", "sql"),
    ]
}


@pytest.fixture
def fixture_dir(tmp_path, monkeypatch):
    (tmp_path / "esco_skills.json").write_text(json.dumps(ESCO_FIXTURE, ensure_ascii=False))
    (tmp_path / "learning_resources.json").write_text(json.dumps(RES_FIXTURE))
    monkeypatch.setattr(tools, "data_dir", lambda: tmp_path)
    return tmp_path


def labels(out):
    return [r["preferred_label"] for r in out["results"]]


# --- normalisation ---------------------------------------------------------------------------


def test_normalize_is_casefold_and_accent_insensitive():
    assert tools.normalize("  Comptabilité  Générale ") == "comptabilite generale"
    assert tools.normalize("C++ / C#") == "c++ c#"
    assert tools.normalize("Next.js") == "next.js"
    assert tools.normalize("gestion d'équipe") == "gestion d equipe"


# --- search_skill_taxonomy -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "language", "expected"),
    [
        ("accounting", "en", "accounting"),
        ("comptabilité", "fr", "comptabilité"),  # French label
        ("COMPTABILITE", "fr", "comptabilité"),  # case + accents
        ("bookkeeping", "en", "accounting"),  # alt label
        ("tenue des comptes", "en", "accounting"),  # French alt label, English output
        ("python", "en", "Python (computer programming)"),  # qualifier stripped
        ("Pyhton", "en", "Python (computer programming)"),  # typo -> difflib
        ("postgres", "en", "PostgreSQL"),  # truncated word
        ("Excel", "fr", "utiliser un tableur"),  # seed alias from the snapshot
    ],
)
async def test_taxonomy_hits(fixture_dir, query, language, expected):
    out = await tools.search_skill_taxonomy(query, language=language)
    assert labels(out)[0] == expected
    assert "ESCO" in out["note"]


async def test_taxonomy_excel_is_not_excellence(fixture_dir):
    out = await tools.search_skill_taxonomy("Excel")
    assert "demand excellence" not in labels(out)


@pytest.mark.parametrize("query", ["Kafka", "Terraform", "zzzz"])
async def test_taxonomy_miss_returns_note(fixture_dir, query):
    out = await tools.search_skill_taxonomy(query)
    assert out["results"] == []
    assert query in out["note"] and "broader" in out["note"]


async def test_taxonomy_empty_query(fixture_dir):
    assert await tools.search_skill_taxonomy("  ") == {"results": [], "note": "empty query"}


async def test_taxonomy_output_is_short(fixture_dir):
    out = await tools.search_skill_taxonomy("accounting", limit=50)
    hit = out["results"][0]
    assert set(hit) == {"uri", "preferred_label", "alt_labels", "description", "related"}
    assert len(hit["description"]) <= 300
    assert len(hit["related"]) == 8
    assert set(hit["related"][0]) == {"uri", "label"}
    assert len(out["results"]) <= tools.MAX_LIMIT


async def test_taxonomy_missing_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "data_dir", lambda: tmp_path / "nowhere")
    out = await tools.search_skill_taxonomy("python")
    assert out == {"results": [], "note": "ESCO snapshot unavailable"}


# --- search_learning_resources ---------------------------------------------------------------


async def test_resources_free_first_then_kind(fixture_dir):
    out = await tools.search_learning_resources("kafka", limit=5)
    titles = [r["title"] for r in out["results"]]
    assert titles == ["kafka-course", "kafka-docs", "kafka-cert"]  # course < docs, paid last
    assert set(out["results"][0]) == {
        "title",
        "url",
        "provider",
        "kind",
        "free",
        "level",
        "language",
        "est_hours",
    }


@pytest.mark.parametrize(
    ("skill", "titles"),
    [
        ("Tests E2E", {"cypress"}),  # alias, case
        ("Apache Kafka", {"kafka-docs", "kafka-course"}),  # alias of one entry -> whole skill
        ("Spark", {"spark", "spark-quickstart"}),  # partial name -> every entry of the skill
        ("SQL", {"sql"}),  # exact skill beats "SQL Server"
    ],
)
async def test_resources_lookup(fixture_dir, skill, titles):
    out = await tools.search_learning_resources(skill, limit=2)
    assert {r["title"] for r in out["results"]} == titles


async def test_resources_limit(fixture_dir):
    out = await tools.search_learning_resources("Kafka", limit=1)
    assert len(out["results"]) == 1 and "1 of 3" in out["note"]


async def test_resources_miss_says_do_not_invent(fixture_dir):
    out = await tools.search_learning_resources("Cobol")
    assert out["results"] == []
    assert "Cobol" in out["note"] and "do not invent" in out["note"]


# --- the real committed data files -----------------------------------------------------------


def test_data_files_are_valid():
    esco = json.loads((CAREER / "esco_skills.json").read_text())
    assert esco["license"] == "CC-BY-4.0" and len(esco["skills"]) > 100
    res = json.loads((CAREER / "learning_resources.json").read_text())["resources"]
    assert len(res) >= 60
    for r in res:
        assert r["url"].startswith("https://") and r["verified_at"]
        assert r["kind"] in {"docs", "course", "tutorial", "certification"}
        assert isinstance(r["free"], bool)
    assert len({r["url"] for r in res}) == len(res)


async def test_real_data_examples():
    assert labels(await tools.search_skill_taxonomy("comptabilité", "fr"))[0] == "comptabilité"
    assert (await tools.search_skill_taxonomy("Kafka"))["results"] == []
    for gap in ("Kafka", "Terraform", "Kubernetes", "dbt", "SAP", "CACES", "Pennylane", "MLflow"):
        assert (await tools.search_learning_resources(gap))["results"], gap


def test_data_dir_default_resolves_to_repo():
    assert tools.data_dir().resolve() == CAREER.resolve()


# --- check_cv_evidence -----------------------------------------------------------------------


@pytest.fixture
def decisions_on(monkeypatch):
    monkeypatch.setattr(get_settings(), "decisions_enabled", True)


async def test_cv_evidence_present(decisions_on, install_fake_decisions):
    fake = install_fake_decisions(lambda state, q: 0.95 if "Spark" in q else 0.05)
    out = await tools.check_cv_evidence("Apache Spark", CV)
    assert out["skill"] == "Apache Spark" and out["present"] is True
    assert 0.9 < out["p_evidence"] <= 1.0
    assert len(fake.questions) == 1 and '"Apache Spark"' in fake.questions[0]


async def test_cv_evidence_absent(decisions_on, install_fake_decisions):
    install_fake_decisions(lambda state, q: 0.05)
    out = await tools.check_cv_evidence("Kafka", CV)
    assert out["present"] is False and out["p_evidence"] < 0.1
    assert "no clear evidence" in out["note"]


async def test_cv_evidence_unavailable(decisions_on):
    # conftest's default decision backend raises (unreachable) -> graceful None
    out = await tools.check_cv_evidence("Kafka", CV)
    assert out["p_evidence"] is None and out["present"] is None
    assert "unavailable" in out["note"]


async def test_cv_evidence_service_raises(decisions_on, monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("bug")

    monkeypatch.setattr("jobmatch.decisions.service.check_skill_evidence", boom)
    out = await tools.check_cv_evidence("Kafka", CV)
    assert out["p_evidence"] is None and "unavailable" in out["note"]


async def test_cv_evidence_disabled(monkeypatch):
    monkeypatch.setattr(get_settings(), "decisions_enabled", False)
    out = await tools.check_cv_evidence("Kafka", CV)
    assert out["p_evidence"] is None and "disabled" in out["note"]


async def test_cv_evidence_missing_inputs(decisions_on):
    assert "missing" in (await tools.check_cv_evidence("", CV))["note"]
    assert "missing" in (await tools.check_cv_evidence("Kafka", "  "))["note"]
