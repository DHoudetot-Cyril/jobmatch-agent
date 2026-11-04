"""Application settings, loaded from environment variables / `.env`.

CONTRACT: every module reads configuration through `get_settings()` only.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM (llama.cpp `llama-server`, OpenAI-compatible) ---
    llm_base_url: str = "http://172.17.0.1:8080/v1"
    llm_model: str = "qwen"
    llm_api_key: str = "not-needed"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 4096
    llm_timeout_s: float = 300.0

    # --- Agent behaviour ---
    max_review_iterations: int = 2  # writer <-> reviewer loop
    min_text_chars_for_pdf: int = 200  # below this a PDF is treated as scanned -> vision OCR

    # --- Decision layer ("System 1", see jobmatch.decisions.service) ---
    # false = exactly the historical LLM-only pipeline (verify_* nodes become no-ops and the
    # reviewer keeps its original prompt/schema). Used as the A/B switch by the evals.
    decisions_enabled: bool = True
    decisions_backend: str = "jevos"
    jevos_url: str = "http://127.0.0.1:8017"  # compose: http://jevos:8017
    decisions_timeout_s: float = 30.0  # per decision HTTP call
    decisions_concurrency: int = 4  # in-flight decisions per event loop
    claim_dismiss_threshold: float = 0.8  # calibrated P(supported) >= this -> issue dismissed
    # Skill fusion v2 (verify_skills): a firm LLM rating is overridden to partial credit (0.5)
    # only on strong disagreement: "full" with P(evidence) < low, "none" with P > high.
    decision_conflict_low: float = 0.2
    decision_conflict_high: float = 0.8

    # --- Career advisor (the one autonomous tool-using agent, graph/teams/career.py) ---
    # false = the career_team is skipped (exact pre-advisor pipeline, no `career_plan`).
    career_advisor_enabled: bool = True
    career_max_tool_rounds: int = 6  # LLM turns that may call tools; then the answer is forced
    career_timeout_s: float = 120.0  # wall-time budget of the tool loop (then: forced answer)
    career_data_dir: str = "data/career"  # local ESCO snapshot + learning-resource catalog

    # --- LangGraph checkpointer (application Postgres, NOT Langfuse's) ---
    database_url: str = "postgresql://jobmatch:jobmatch@localhost:5433/jobmatch"

    # --- Langfuse (self-hosted) ---
    langfuse_enabled: bool = True
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = "pk-lf-jobmatch-local"
    langfuse_secret_key: str = "sk-lf-jobmatch-local"

    # --- API ---
    max_upload_mb: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()
