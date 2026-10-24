from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
DEFAULT_ENV_FILE = BACKEND_DIR / ".env"
REPO_ENV_FILE = REPO_ROOT / ".env"
DEFAULT_CSV_PATH = REPO_ROOT / "data" / "bangla_products_5k.csv"
DEFAULT_DATABASE_URL = "sqlite:////tmp/rag-system.db"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(DEFAULT_ENV_FILE, REPO_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = DEFAULT_DATABASE_URL

    # Groq
    groq_api_key: str | None = None
    groq_model: str = "llama-3.1-8b-instant"

    # App
    session_ttl_seconds: int = 1800  # 30 minutes
    max_sessions: int = 5000
    max_retrieval_results: int = 5
    csv_path: str = str(DEFAULT_CSV_PATH)

    # Abuse protection for the LLM-backed endpoints
    api_token: str | None = None
    rate_limit_per_minute: int = 30
    max_concurrent_llm_calls: int = 4

    @field_validator("csv_path", mode="before")
    @classmethod
    def _resolve_csv_path(cls, value: str | Path) -> str:
        path = Path(value)
        if not path.is_absolute():
            path = (BACKEND_DIR / path).resolve()
        return str(path)


settings = Settings()
