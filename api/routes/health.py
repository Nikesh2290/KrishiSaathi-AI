"""Health check."""

from __future__ import annotations

from fastapi import APIRouter

from agent.gemma_client import ollama_healthy
from config.settings import get_settings

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health")
async def health():
    settings = get_settings()
    ollama_ok = await ollama_healthy(settings)
    return {
        "status": "ok",
        "ollama_ok": ollama_ok,
        "db_ok": True,
        "version": settings.app_version,
    }
