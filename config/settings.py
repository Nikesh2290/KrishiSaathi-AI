"""Application settings (env-driven, hackathon free-tier defaults)."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


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

    # Supabase (Auth + Postgres + pgvector). Anon key is used for email/password auth;
    # service role is required for server-side DB/RPC (bypasses RLS).
    supabase_url: str = Field(default="", alias="SUPABASE_URL")
    supabase_anon_key: str = Field(default="", alias="SUPABASE_ANON_KEY")
    supabase_service_role_key: str = Field(
        default="",
        alias="SUPABASE_SERVICE_ROLE_KEY",
        description="Server-only key for farmer_twin, query_history, scheme_vectors sync/search.",
    )

    database_path: str = Field(default="./data/krishisaathi.db", alias="DATABASE_PATH")
    chroma_path: str = Field(default="./data/chroma", alias="CHROMA_PATH")

    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        alias="CORS_ORIGINS",
    )

    max_react_iterations: int = Field(default=6, alias="MAX_REACT_ITERATIONS")
    tool_timeout_seconds: float = Field(
        default=15.0,
        alias="TOOL_TIMEOUT_SECONDS",
        description="Fallback cap for unknown tools only (known tools use per-tool timeouts below).",
    )
    vision_timeout_seconds: float = Field(
        default=60.0,
        alias="VISION_TIMEOUT_SECONDS",
        description="Cap for multimodal disease detection (AI Studio / Ollama vision).",
    )
    llm_tool_timeout_seconds: float = Field(
        default=30.0,
        alias="LLM_TOOL_TIMEOUT_SECONDS",
        description="Cap for tools that end with a text LLM call (scheme, crop_planner, financial).",
    )
    climate_timeout_seconds: float = Field(
        default=12.0,
        alias="CLIMATE_TIMEOUT_SECONDS",
        description="Cap for live weather HTTP fetch (Open-Meteo).",
    )
    weather_cache_ttl_seconds: int = Field(
        default=86400,
        alias="WEATHER_CACHE_TTL_SECONDS",
        description="TTL for home-screen weather cache in SQLite (seconds). Default 24 h.",
    )
    io_tool_timeout_seconds: float = Field(
        default=5.0,
        alias="IO_TOOL_TIMEOUT_SECONDS",
        description="Cap for fast local I/O (offline climate JSON, mandi CSV in thread).",
    )

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_json: bool = Field(default=True, alias="LOG_JSON")
    log_file: str = Field(default="", alias="LOG_FILE")

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

    @property
    def supabase_auth_configured(self) -> bool:
        return bool(self.supabase_url and self.supabase_anon_key)

    @property
    def supabase_db_configured(self) -> bool:
        """DB + vector RPC from this API (prefers service role)."""
        return bool(
            self.supabase_url
            and (self.supabase_service_role_key or self.supabase_anon_key)
        )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    ssl_cert_file = os.environ.get("SSL_CERT_FILE", "").strip()
    if ssl_cert_file and not Path(ssl_cert_file).is_file():
        logger.warning(
            "Ignoring invalid SSL_CERT_FILE path: %s (falling back to default trust store)",
            ssl_cert_file,
        )
        os.environ.pop("SSL_CERT_FILE", None)
    return settings
