"""Cover-letter length is checked in code, not estimated by the LLM reviewer. No LLM needed.

Regression for a real run (docs/demo/run.json): the reviewer called a 281-word letter "about
230 words" and flagged 335 words as "below the 250-400 range"; the rewrite that followed added
unsupported claims (1 issue -> 10).
"""

import pytest

from jobmatch.config import get_settings
from jobmatch.graph.teams.writing import count_words, length_issue, review_letter
from jobmatch.schemas import CoverLetter, Review, ReviewerOutput


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Madame, Monsieur,", 2),
        ("d’expérience et d'Airflow", 3),  # an elided article is part of its word
        ("Python — SQL - Spark", 3),  # lone dashes are not words
        ("15 millions d'événements", 3),
        ("  \n\n ", 0),
    ],
)
def test_count_words(text, expected):
    assert count_words(text) == expected


def test_length_issue_bounds():
    words = lambda n: "mot " * n  # noqa: E731
    assert length_issue(words(250), 250, 400) is None
    assert length_issue(words(400), 250, 400) is None
    short = length_issue(words(230), 250, 400)
    assert "too short: 230 words" in short and "Add about 20 words" in short
    long = length_issue(words(410), 250, 400)
    assert "too long: 410 words" in long and "Remove about 10 words" in long


def _state(offer, profile, body: str) -> dict:
    return {
        "offer": offer,
        "profile": profile,
        "cv_text": "CV ...",
        "cover_letter": CoverLetter(subject="Candidature", body=body),
    }


@pytest.fixture(params=[True, False], ids=["decisions_on", "decisions_off"])
def reviewer_schema(request, monkeypatch):
    """Both reviewer paths: claims prompt (`ReviewerOutput`) and historical one (`Review`)."""
    monkeypatch.setattr(get_settings(), "decisions_enabled", request.param)
    return ReviewerOutput if request.param else Review


async def test_wrong_llm_length_estimate_is_dropped(
    install_fake_llm, offer, profile, letter, reviewer_schema
):
    # the exact issue the reviewer produced on a 335-word letter
    wrong = "The letter is too short: it contains about 310 words, below the required 250-400."
    install_fake_llm({Review: [reviewer_schema(approved=False, issues=[wrong])]})
    out = await review_letter(_state(offer, profile, letter.body))
    review = out["review"]
    assert review.issues == [] and review.approved is True


async def test_short_letter_is_rejected_with_exact_count(
    install_fake_llm, offer, profile, reviewer_schema
):
    install_fake_llm({Review: [reviewer_schema(approved=True)]})  # the LLM misses it
    out = await review_letter(_state(offer, profile, "Madame, Monsieur, " + "mot " * 98))
    review = out["review"]
    assert review.approved is False
    assert review.issues == [
        "The letter is too short: 100 words (counted in code), the target is 250-400. "
        "Add about 150 words of relevant content taken from the CV, without inventing facts."
    ]


async def test_other_issues_and_claims_are_kept(install_fake_llm, offer, profile, letter):
    claim = "j'ai 10 ans d'expérience sur Kafka"
    install_fake_llm(
        {
            Review: [
                ReviewerOutput(
                    approved=False,
                    issues=["The company name is missing.", "Too long for the offer: 900 words"],
                    unsupported_claims=[claim],
                )
            ]
        }
    )
    out = await review_letter(_state(offer, profile, letter.body))
    issues = out["review"].issues
    assert "The company name is missing." in issues
    assert any(claim in i for i in issues)
    assert not any("900 words" in i for i in issues)
    assert out["review"].approved is False


async def test_writer_prompt_uses_configured_bounds(
    install_fake_llm, offer, profile, letter, monkeypatch
):
    from jobmatch.graph.teams.writing import write_letter
    from jobmatch.schemas import MatchResult

    monkeypatch.setattr(get_settings(), "letter_min_words", 180)
    monkeypatch.setattr(get_settings(), "letter_max_words", 300)
    fake = install_fake_llm({CoverLetter: [letter]})
    match = MatchResult(score=70, matched_skills=[], gaps=[], strengths=[], summary="ok")
    await write_letter({"offer": offer, "profile": profile, "match": match})
    assert "180-300 words" in fake.calls["CoverLetter"][0][0].content
