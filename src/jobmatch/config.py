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
