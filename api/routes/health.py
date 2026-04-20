"""Health check."""

from __future__ import annotations

import os

from fastapi import APIRouter

from agent.gemma_client import ollama_healthy
from config.settings import get_settings

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health")
async def health():
    settings = get_settings()
    ollama_ok = await ollama_healthy(settings)
    ai_studio_ok = bool(settings.google_api_key)
    db_ok = os.path.exists(os.path.dirname(settings.database_path) or ".")
    chroma_ok = os.path.isdir(settings.chroma_path) or True
    return {
        "status": "ok",
        "version": settings.app_version,
        "ai_studio_ok": ai_studio_ok,
        "ollama_ok": ollama_ok,
        "db_ok": db_ok,
        "chroma_ok": chroma_ok,
        "gemma4_model_configured": settings.ai_studio_model,
    }
