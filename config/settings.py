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
    app_version: str = "0.1.0"

    # Google AI Studio (Gemini API) — Gemma 4 via langchain-google-genai
    google_api_key: str = Field(default="", alias="GOOGLE_AI_STUDIO_KEY")
    ai_studio_model: str = Field(
        default="gemma-3-27b-it",
        alias="AI_STUDIO_MODEL",
        description="Gemma model id on Gemini API (override to gemma-4-* when available in your region)",
    )

    # Ollama (local Gemma)
    ollama_base_url: str = Field(default="http://127.0.0.1:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="gemma2:2b", alias="OLLAMA_MODEL")

    # Routing: auto | local | cloud
    connectivity_mode: str = Field(default="auto", alias="CONNECTIVITY_MODE")

    # SQLite
    database_path: str = Field(default="./data/krishisaathi.db", alias="DATABASE_PATH")

    # Chroma
    chroma_path: str = Field(default="./data/chroma", alias="CHROMA_PATH")

    # CORS — comma-separated origins for frontend
    cors_origins: str = Field(default="http://localhost:3000,http://localhost:5173", alias="CORS_ORIGINS")

    # Agent limits
    max_react_iterations: int = Field(default=6, alias="MAX_REACT_ITERATIONS")
    tool_timeout_seconds: float = Field(default=8.0, alias="TOOL_TIMEOUT_SECONDS")

    # Rate limit (per farmer_id, per minute)
    rate_limit_per_minute: int = Field(default=10, alias="RATE_LIMIT_PER_MINUTE")

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

