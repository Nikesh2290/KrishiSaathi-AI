"""Application settings (env-driven, hackathon free-tier defaults)."""

from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "KrishiSaathi AI"
    app_version: str = "0.2.0"

    # Google AI Studio — Gemma 4 family
    google_api_key: str = Field(default="", alias="GOOGLE_AI_STUDIO_KEY")
    ai_studio_model: str = Field(
        default="gemma-4-26b-a4b-it",
        alias="AI_STUDIO_MODEL",
        description="Primary Gemma 4 model on AI Studio (fast, MoE).",
    )
    ai_studio_model_heavy: str = Field(
        default="gemma-4-31b-it",
        alias="AI_STUDIO_MODEL_HEAVY",
        description="Gemma 4 heavy model used for escalation when confidence low.",
    )

    # Ollama — server-side dev convenience only; production path is AI Studio
    ollama_base_url: str = Field(default="http://127.0.0.1:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="gemma-4-e4b-it", alias="OLLAMA_MODEL")

    connectivity_mode: str = Field(default="auto", alias="CONNECTIVITY_MODE")

    database_path: str = Field(default="./data/krishisaathi.db", alias="DATABASE_PATH")
    chroma_path: str = Field(default="./data/chroma", alias="CHROMA_PATH")

    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        alias="CORS_ORIGINS",
    )

    max_react_iterations: int = Field(default=6, alias="MAX_REACT_ITERATIONS")
    tool_timeout_seconds: float = Field(default=8.0, alias="TOOL_TIMEOUT_SECONDS")

    # Sync + image + confidence thresholds
    sync_bundle_cache_ttl_seconds: int = Field(
        default=3600, alias="SYNC_BUNDLE_CACHE_TTL_SECONDS"
    )
    image_upload_ttl_seconds: int = Field(default=3600, alias="IMAGE_UPLOAD_TTL_SECONDS")
    image_max_mb: int = Field(default=5, alias="IMAGE_MAX_MB")
    confidence_threshold_low: float = Field(
        default=0.70, alias="CONFIDENCE_THRESHOLD_LOW"
    )

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
