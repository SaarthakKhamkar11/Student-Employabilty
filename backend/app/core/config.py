"""Application settings, loaded from environment variables or the repo-root .env.

Secrets are typed SecretStr so they never appear in repr() or logs, and none
of them has a default. Copy .env.example to .env (gitignored) to configure.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"

    database_url: SecretStr | None = None  # contains the database password
    redis_url: str | None = None
    jwt_secret: SecretStr | None = None

    # LLM: project decision 2026-10-08 is Gemini. Both key and model must be set,
    # otherwise the Supervisor routes deterministically (spec §22).
    llm_provider: Literal["gemini"] = "gemini"
    gemini_api_key: SecretStr | None = None
    gemini_model: str | None = None
    llm_timeout_seconds: float = Field(10.0, gt=0, le=60)
    llm_max_retries: int = Field(1, ge=0, le=1)  # Phase 3 rule: at most one retry
    supervisor_timeout_seconds: float = Field(300.0, gt=0)

    # Embeddings: the interim TF-IDF baseline until Phase 6 selects a model from evidence (docs/adr/0004).
    embedding_model: Literal["tfidf_baseline"] = "tfidf_baseline"
    processed_data_dir: Path = REPO_ROOT / "data" / "processed"

    # Owner decision (docs/adr/0009): hide postings whose stated gender preference excludes the student.
    # Unknown student gender is never hidden. Set JOB_GENDER_VISIBILITY_FILTER=false to show every posting.
    job_gender_visibility_filter: bool = True

    # Agent runtime limits (BaseAgent). Every agent run terminates within these.
    agent_max_retries: int = Field(2, ge=0, le=10)
    agent_max_steps: int = Field(10, ge=1, le=100)
    agent_timeout_seconds: float = Field(60.0, gt=0)
    tool_timeout_seconds: float = Field(30.0, gt=0)
    tool_retry_backoff_seconds: float = Field(0.5, ge=0)

    @property
    def llm_configured(self) -> bool:
        return bool(self.gemini_api_key and self.gemini_api_key.get_secret_value() and self.gemini_model)

    @model_validator(mode="after")
    def _production_requires_secrets(self) -> "Settings":
        if self.environment == "production" and not (self.jwt_secret and self.jwt_secret.get_secret_value()):
            raise ValueError("JWT_SECRET must be set in production")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
