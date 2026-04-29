"""Offline sync bundle endpoint + Supabase push."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, Query, Response

from config.settings import get_settings
from offline.bundle_builder import build_gzip_bundle

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["sync"])


_cache: Dict[str, Tuple[bytes, str, int]] = {}


def _cache_key(state: str, district: str) -> str:
    return f"{state.lower()}|{district.lower()}"


def _get_or_build(state: str, district: str, ttl: int) -> Tuple[bytes, str]:
    key = _cache_key(state, district)
    now = int(time.time())
    hit = _cache.get(key)
    if hit and hit[2] > now:
        return hit[0], hit[1]
    raw, version = build_gzip_bundle(state, district)
    _cache[key] = (raw, version, now + ttl)
    return raw, version


@router.get("/sync/bundle")
async def get_sync_bundle(
    state: str = Query(..., min_length=1),
    district: str = Query(..., min_length=1),
    bundle_version: Optional[str] = Query(None),
) -> Response:
    settings = get_settings()
    raw, version = _get_or_build(state, district, settings.sync_bundle_cache_ttl_seconds)
    if bundle_version and bundle_version == version:
        return Response(status_code=304)
    return Response(
        content=raw,
        media_type="application/json",
        headers={
            "Content-Encoding": "gzip",
            "X-Bundle-Version": version,
            "Cache-Control": f"public, max-age={settings.sync_bundle_cache_ttl_seconds}",
        },
    )


@router.post("/sync/push")
async def push_to_supabase() -> Dict[str, Any]:
    """Push unsynced SQLite farmer rows, query logs, and scheme vectors to Supabase."""
    settings = get_settings()
    if not settings.supabase_db_configured:
        return {"ok": False, "skipped": True, "reason": "Supabase DB not configured"}
    from offline.supabase_sync import SupabaseSync

    try:
        return await SupabaseSync(settings).run()
    except Exception:
        logger.exception("SupabaseSync failed")
        raise
