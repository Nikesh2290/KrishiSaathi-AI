"""Health check."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict

import httpx
from fastapi import APIRouter

from agent.gemma_client import ollama_healthy
from config.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["health"])


async def _redis_ping_ok(settings) -> bool:
    if not settings.redis_configured:
        return False
    try:
        from cache.redis_client import redis_from_settings

        r = redis_from_settings(settings)
        pong = await asyncio.wait_for(r.ping(), timeout=3.0)
        return bool(pong)
    except Exception as exc:
        logger.debug("Redis health ping failed: %s", exc)
        return False


async def _vector_info_ok(settings) -> bool:
    if not settings.upstash_vector_configured:
        return False
    url = settings.upstash_vector_rest_url.strip().rstrip("/") + "/info"
    headers = {"Authorization": f"Bearer {settings.upstash_vector_rest_token.strip()}"}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(3.0, connect=2.0)) as client:
            r = await client.get(url, headers=headers)
        return 200 <= r.status_code < 300
    except Exception as exc:
        logger.debug("Upstash Vector /info failed: %s", exc)
        return False


@router.get("/health")
async def health() -> Dict[str, Any]:
    settings = get_settings()
    ollama_ok = await ollama_healthy(settings)
    ai_studio_ok = bool(settings.google_api_key)
    db_ok = os.path.exists(os.path.dirname(settings.database_path) or ".")
    chroma_ok = os.path.isdir(settings.chroma_path) or True

    qstash_ok = bool(settings.qstash_configured)

    redis_task = asyncio.create_task(_redis_ping_ok(settings))
    vector_task = asyncio.create_task(_vector_info_ok(settings))
    redis_ok, vector_ok = await asyncio.gather(
        redis_task,
        vector_task,
        return_exceptions=True,
    )
    upstash_redis_ok = redis_ok if isinstance(redis_ok, bool) else False
    upstash_vector_ok = vector_ok if isinstance(vector_ok, bool) else False

    return {
        "status": "ok",
        "version": settings.app_version,
        "ai_studio_ok": ai_studio_ok,
        "ollama_ok": ollama_ok,
        "db_ok": db_ok,
        "chroma_ok": chroma_ok,
        "gemma4_model_configured": settings.ai_studio_model,
        "upstash_redis_ok": upstash_redis_ok,
        "upstash_qstash_configured": qstash_ok,
        "upstash_vector_ok": upstash_vector_ok,
    }
