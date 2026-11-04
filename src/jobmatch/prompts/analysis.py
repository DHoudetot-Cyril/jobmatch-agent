"""Prompts of the Analysis team (offer parser + CV parser)."""

from langchain_core.prompts import ChatPromptTemplate

_EXTRACTION_RULES = """Rules:
- Extract ONLY information explicitly present in the document. Never invent or infer facts
  that are not written (no made-up responsibilities, skills, companies or dates).
- Keep the wording of the document (same language) for free-text fields.
- Use short canonical names for skills (e.g. "Python", "Apache Spark", "Docker").
- Leave a field empty / null when the document does not provide the information."""

PARSE_OFFER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert technical recruiter. You turn a job offer into structured data.\n"
            + _EXTRACTION_RULES
            + """
- Classify each skill: "required" when the offer presents it as mandatory (e.g. "required",
  "must have", "indispensable", "maîtrise de", listed in the main requirements);
  "nice_to_have" when it is optional (e.g. "a plus", "idéalement", "serait un plus", "bonus").
- `seniority` is one of: junior, mid, senior, lead (infer from years of experience if stated:
  0-2 junior, 3-5 mid, 6-9 senior, 10+ lead).
- `language` is the ISO 639-1 code of the offer's language (e.g. "fr", "en").""",
        ),
        ("human", "Job offer:\n<<<\n{offer_text}\n>>>"),
    ]
)

PARSE_CV_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert technical recruiter. You turn a CV (resume) into a structured "
            "candidate profile.\n"
            + _EXTRACTION_RULES
            + """
- `skills`: every technical skill / tool mentioned in the CV (skills section AND experiences).
- `experiences`: one entry per position, most recent first; `duration_months` computed from
  the dates when possible; `highlights` = the concrete achievements as written.
- `education`: one string per degree; `languages`: spoken languages with level if given.""",
        ),
        ("human", "CV:\n<<<\n{cv_text}\n>>>"),
    ]
)
