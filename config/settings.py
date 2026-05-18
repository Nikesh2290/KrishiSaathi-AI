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
        description="Primary Gemma 4 model on AI Studio (chat: 26B MoE).",
    )
    ai_studio_voice_model: str = Field(
        default="gemma-4-e4b-it",
        alias="AI_STUDIO_VOICE_MODEL",
        description="Gemma 4 model for voice mode on AI Studio (E4B, faster for real-time).",
    )
    ai_studio_model_heavy: str = Field(
        default="gemma-4-31b-it",
        alias="AI_STUDIO_MODEL_HEAVY",
        description="Gemma 4 heavy model used for escalation when confidence low.",
    )

    # Ollama — server-side dev convenience only; production path is AI Studio
    ollama_base_url: str = Field(default="http://127.0.0.1:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(
        default="gemma-4-e4b-it",
        alias="OLLAMA_MODEL",
        description="Ollama model for voice mode (fast, E4B).",
    )
    ollama_chat_model: str = Field(
        default="gemma-4-26b-it",
        alias="OLLAMA_CHAT_MODEL",
        description="Ollama model for chat mode (larger, more capable).",
    )

    connectivity_mode: str = Field(default="auto", alias="CONNECTIVITY_MODE")

    # Supabase (Auth + Postgres + pgvector). Anon key is used for email/password auth;
    # service role is used for server-side DB/RPC (elevated access when RLS is enabled).
    supabase_url: str = Field(default="", alias="SUPABASE_URL")
    supabase_anon_key: str = Field(default="", alias="SUPABASE_ANON_KEY")
    supabase_service_role_key: str = Field(
        default="",
        alias="SUPABASE_SERVICE_ROLE_KEY",
        description="Server-only key for farmer_twin, conversation_metadata, query_history, scheme_vectors sync/search.",
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

    # data.gov.in — mandi (OGD) API for sync + market tool fallback
    ogd_api_key: str = Field(default="", alias="OGD_API_KEY")
    mandi_price_ttl_seconds: int = Field(
        default=86400,
        alias="MANDI_PRICE_TTL_SECONDS",
        description="How long synced mandi prices stay valid in local DB (default 24 h).",
    )
    market_tool_timeout_seconds: float = Field(
        default=60.0,
        alias="MARKET_TOOL_TIMEOUT_SECONDS",
        description="Read/connect budget for mandi OGD HTTP (sync + tool fallback).",
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

    # LiveKit — voice room join tokens (API) + worker env (same vars)
    livekit_url: str = Field(default="", alias="LIVEKIT_URL")
    livekit_api_key: str = Field(default="", alias="LIVEKIT_API_KEY")
    livekit_api_secret: str = Field(default="", alias="LIVEKIT_API_SECRET")
    voice_token_ttl_seconds: int = Field(
        default=3600,
        alias="VOICE_TOKEN_TTL_SECONDS",
        description="JWT lifetime for LiveKit participant tokens (seconds).",
    )

    # Voice worker calls this Krishi HTTP API (compose: http://api:7860)
    krishi_api_base_url: str = Field(
        default="http://127.0.0.1:8000",
        alias="KRISHI_API_BASE_URL",
        description="Base URL for POST /api/v1/query/stream from the LiveKit voice worker.",
    )

    # Deepgram STT + TTS (voice worker; plugins also read DEEPGRAM_API_KEY from env)
    deepgram_api_key: str = Field(default="", alias="DEEPGRAM_API_KEY")

    # Dev fallback when participant JWT metadata is missing
    voice_default_farmer_id: str = Field(
        default="",
        alias="VOICE_DEFAULT_FARMER_ID",
        description="Optional UUID string; voice worker uses this if join token had no farmer metadata.",
    )

    # --- Upstash Redis (REST; HF Space + Railway voice worker reads warm cache when configured)
    upstash_redis_rest_url: str = Field(
        default="",
        alias="UPSTASH_REDIS_REST_URL",
        description="Upstash Redis REST URL (shown as UPSTASH_REDIS_REST_URL in console).",
    )
    upstash_redis_rest_token: str = Field(
        default="",
        alias="UPSTASH_REDIS_REST_TOKEN",
        description="Upstash Redis REST token.",
    )
    # Aliases matching common env names from older docs / plan
    upstash_redis_url: str = Field(default="", alias="UPSTASH_REDIS_URL")
    upstash_redis_token: str = Field(default="", alias="UPSTASH_REDIS_TOKEN")

    # TTLs for Redis-backed cache (defaults: conservative preset C from plan doc)
    redis_twin_ttl_seconds: int = Field(
        default=86400, alias="REDIS_TWIN_TTL_SECONDS", description="Default 24h."
    )
    redis_context_packet_ttl_seconds: int = Field(
        default=604800, alias="REDIS_CONTEXT_PACKET_TTL_SECONDS", description="Default 7d."
    )
    redis_session_ttl_seconds: int = Field(
        default=86400, alias="REDIS_SESSION_TTL_SECONDS", description="Default 24h."
    )
    redis_weather_ttl_seconds: int = Field(
        default=3600, alias="REDIS_WEATHER_TTL_SECONDS", description="Default 1h."
    )
    redis_mandi_ttl_seconds: int = Field(
        default=14400, alias="REDIS_MANDI_TTL_SECONDS", description="Default 4h."
    )
    redis_schemes_index_ttl_seconds: int = Field(
        default=2592000, alias="REDIS_SCHEMES_INDEX_TTL_SECONDS", description="Default 30d."
    )
    redis_conversations_list_ttl_seconds: int = Field(
        default=604800,
        alias="REDIS_CONVERSATIONS_LIST_TTL_SECONDS",
        description="TTL for per-farmer conversation list cache in Redis (default 7d).",
    )

    # QStash — async jobs (persist to Supabase, sync, bundle regenerate, optional escalation enqueue)
    qstash_token: str = Field(default="", alias="QSTASH_TOKEN")
    qstash_url: str = Field(
        default="",
        alias="QSTASH_URL",
        description="Usually https://qstash.upstash.io (default for SDK if empty).",
    )
    qstash_current_signing_key: str = Field(default="", alias="QSTASH_CURRENT_SIGNING_KEY")
    qstash_next_signing_key: str = Field(default="", alias="QSTASH_NEXT_SIGNING_KEY")
    public_api_base_url: str = Field(
        default="",
        alias="PUBLIC_API_BASE_URL",
        description="Public HTTPS base of this API (no trailing slash). Used for QStash callbacks to /api/v1/internal/qstash/...",
    )

    warmup_secret: str = Field(default="", alias="WARMUP_SECRET")
    warmup_states: str = Field(
        default="",
        alias="WARMUP_STATES",
        description="Comma-separated state names used by warmup cron (e.g. Maharashtra,Punjab).",
    )
    warmup_districts: str = Field(
        default="",
        alias="WARMUP_DISTRICTS",
        description="Comma-separated districts paired with warmup states or used as list.",
    )
    warmup_coordinates: str = Field(
        default="",
        alias="WARMUP_COORDINATES",
        description="Weather warm-up coords: LAT:LNG pairs separated by | (e.g. 26.9124:75.7873|19.0760:72.8777).",
    )

    google_embedding_model: str = Field(
        default="text-embedding-004",
        alias="GOOGLE_EMBEDDING_MODEL",
        description="Gemini embeddings for Upstash Vector (768 dims for text-embedding-004).",
    )

    upstash_vector_rest_url: str = Field(default="", alias="UPSTASH_VECTOR_REST_URL")
    upstash_vector_rest_token: str = Field(default="", alias="UPSTASH_VECTOR_REST_TOKEN")

    startup_cache_warmup: bool = Field(
        default=True,
        alias="STARTUP_CACHE_WARMUP",
        description="If true and Redis is configured, enqueue / run warmup after startup.",
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

    @property
    def livekit_configured(self) -> bool:
        return bool(self.livekit_url and self.livekit_api_key and self.livekit_api_secret)

    @property
    def redis_rest_url(self) -> str:
        return (self.upstash_redis_rest_url or self.upstash_redis_url or "").strip()

    @property
    def redis_rest_token(self) -> str:
        return (self.upstash_redis_rest_token or self.upstash_redis_token or "").strip()

    @property
    def redis_configured(self) -> bool:
        return bool(self.redis_rest_url and self.redis_rest_token)

    @property
    def qstash_configured(self) -> bool:
        return bool(
            self.qstash_token.strip()
            and self.public_api_base_url.strip()
            and self.qstash_current_signing_key.strip()
            and self.qstash_next_signing_key.strip()
        )

    @property
    def upstash_vector_configured(self) -> bool:
        return bool(
            self.upstash_vector_rest_url.strip() and self.upstash_vector_rest_token.strip()
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
