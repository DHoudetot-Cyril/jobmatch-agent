"""Prompts of the Matching team (scorer + gap analyst)."""

from langchain_core.prompts import ChatPromptTemplate

SCORE_MATCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a rigorous technical recruiter assessing how well a candidate fits a job.

Assess EVERY skill listed in the job offer, one entry per offer skill (keep its exact name
and level). For each one decide `match`:
- "full": the CV clearly shows the skill (listed or used in an experience);
- "partial": a close equivalent or only superficial/old exposure (e.g. Azure instead of AWS,
  a school project only);
- "none": no evidence in the CV.
`evidence` must quote or paraphrase the CV passage that supports the decision, or say what is
missing. Never assume a skill the CV does not mention.

Then rate `experience_fit` from 0 to 100: how well the candidate's years of experience,
seniority and past responsibilities match the offer's seniority and responsibilities.

Scoring rubric (the final score is computed by code from your assessment, so be accurate):
- required skills: 70% of the score (full = 1, partial = 0.5, none = 0);
- nice-to-have skills: 15% of the score;
- experience fit: 15% of the score.
Required skills therefore weigh much more than nice-to-have ones.

`strengths`: 2-4 concrete strengths of the candidate FOR THIS JOB, grounded in the CV.
`summary`: 2-3 sentences justifying the fit, written in the offer's language ({language}).""",
        ),
        (
            "human",
            "Job offer (structured):\n{offer_json}\n\nCandidate profile (structured):\n"
            "{profile_json}",
        ),
    ]
)

ANALYZE_GAPS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a career coach. For each skill gap below (offer skills that the candidate
covers only partially or not at all), give concrete, actionable `advice` to close or mitigate
the gap: a specific learning resource or project, how to leverage related experience from the
CV, or how to address it honestly in the application/interview.
Keep the skill name and level exactly as given. One entry per gap, in the same order.
Write the advice in the offer's language ({language}), 1-2 sentences each.""",
        ),
        (
            "human",
            "Job offer:\n{offer_json}\n\nCandidate profile:\n{profile_json}\n\n"
            "Gaps to address:\n{gaps}",
        ),
    ]
)
