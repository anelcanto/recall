import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

RECALL_DIR = Path.home() / ".recall"

_env_file = os.environ.get(
    "RECALL_ENV_FILE",
    str(RECALL_DIR / ".env"),
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_env_file if os.path.isfile(_env_file) else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "memories"

    ollama_base_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    ollama_embed_path: str = "/api/embed"

    api_host: str = "127.0.0.1"
    api_port: int = 8100
    api_auth_token: str = ""

    max_text_length: int = 8000
    max_batch_size: int = 100
    health_check_timeout_s: float = 5.0


settings = Settings()
