"""Singleton async Upstash Redis client."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from config.settings import Settings, get_settings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_redis: Any | None = None


def redis_from_settings(settings: Settings) -> Any | None:
    """Instantiate Redis async client (caller must check redis_configured)."""
    global _redis
    if _redis is None:
        from upstash_redis.asyncio import Redis

        _redis = Redis(
            settings.redis_rest_url,
            settings.redis_rest_token,
        )
    return _redis


async def close_redis() -> None:
    global _redis
    _redis = None


async def json_get_maybe(r: Any, key: str) -> Optional[Any]:
    try:
        raw = await r.get(key)
        if raw is None:
            return None
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="replace")
        import json

        return json.loads(raw)
    except Exception as e:
        logger.debug("Redis get/decode miss for %s: %s", key, e)
        return None


async def json_setex(r: Any, key: str, ttl_seconds: int, obj: Any) -> None:
    import json

    payload = json.dumps(obj, ensure_ascii=False, default=str)
    await r.set(key, payload, ex=ttl_seconds)


def get_redis_for_request(settings: Optional[Settings] = None) -> Any | None:
    """Return singleton Redis client if configured, else None."""
    s = settings or get_settings()
    if not s.redis_configured:
        return None
    return redis_from_settings(s)
