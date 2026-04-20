"""Offline sync bundle endpoint."""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

from fastapi import APIRouter, Query, Response

from config.settings import get_settings
from offline.bundle_builder import build_gzip_bundle

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
