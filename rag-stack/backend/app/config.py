from __future__ import annotations

import functools
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR.parent / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(BASE_DIR.parent.parent / ".env"), env_file_encoding="utf-8", case_sensitive=False)

    anthropic_api_key: str = Field(alias="ANTHROPIC_API_KEY")
    claude_model: str = Field(default="claude-3-5-sonnet-20241022", alias="CLAUDE_MODEL")
    embeddings_provider: str = Field(default="sentence-transformers", alias="EMBEDDINGS_PROVIDER")
    embeddings_model: str = Field(default="BAAI/bge-m3", alias="EMBEDDINGS_MODEL")
    database_url: str = Field(alias="DATABASE_URL")
    api_key: str = Field(alias="API_KEY")
    enable_rerank: bool = Field(default=False, alias="ENABLE_RERANK")
    enable_hybrid: bool = Field(default=False, alias="ENABLE_HYBRID")
    max_tokens: int = Field(default=1024, alias="MAX_TOKENS")
    temperature: float = Field(default=0.0, alias="TEMPERATURE")
    response_language: str = Field(default="es", alias="RESPONSE_LANGUAGE")
    rate_limit_per_minute: int = Field(default=60, alias="RATE_LIMIT_PER_MINUTE")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    ingest_default_repo: str = "local"
    ingest_default_tag: str = "local"
    ingest_default_acl: List[str] = Field(default_factory=lambda: ["public"])

    @property
    def sync_database_url(self) -> str:
        if "+psycopg_async" in self.database_url:
            return self.database_url.replace("+psycopg_async", "+psycopg")
        if "+asyncpg" in self.database_url:
            return self.database_url.replace("+asyncpg", "+psycopg")
        return self.database_url


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
