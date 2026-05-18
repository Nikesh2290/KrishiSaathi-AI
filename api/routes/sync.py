"""Offline sync bundle endpoint + Supabase push."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, Query, Response

from config.settings import get_settings
from cache import cache_keys as ck
from cache import context_builder, redis_client
from offline.bundle_builder import build_gzip_bundle

from db.persistence import resolve_farmer_twin


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["sync"])


_cache: Dict[str, Tuple[bytes, str, int]] = {}


def invalidate_bundle_cache_all() -> None:
    """Invalidate in-process gz bundle cache after warm-up."""
    global _cache
    _cache.clear()


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
    farmer_id: Optional[str] = Query(None),
    conversation_id: Optional[str] = Query(None),
) -> Response:
    settings = get_settings()

    fid = (farmer_id or "").strip()
    cid = (conversation_id or "").strip()
    try:
        if settings.redis_configured and fid:
            conn = redis_client.redis_from_settings(settings)

            twin = await resolve_farmer_twin(fid, connectivity="online", settings=settings)
            if twin is not None:
                await context_builder.cache_farmer_twin(conn, twin, settings)

            if twin:
                lt = getattr(twin, "location", None)
                if lt and getattr(lt, "lat", None) and getattr(lt, "lng", None):
                    try:
                        wd = await _maybe_import_climate_engine().get_weather_widget(
                            float(lt.lat), float(lt.lng), settings
                        )
                        await redis_client.json_setex(
                            conn,
                            ck.weather_key(float(lt.lat), float(lt.lng)),
                            ck.ttl_weather(settings),
                            wd,
                        )
                    except Exception:
                        logger.warning("Weather cache update on bundle failed", exc_info=True)

                st_pair = getattr(lt, "state", None) or ""
                dist_pair = getattr(lt, "district", None) or ""
                if st_pair.strip() or dist_pair.strip():
                    await _warm_mandi_to_redis(
                        conn,
                        (st_pair or state).strip(),
                        (dist_pair or district).strip(),
                        settings,
                    )

            if fid and cid:
                await context_builder.rebuild_context_packet(conn, fid, cid, twin, settings)
    except Exception:
        logger.exception("Redis hydrate on bundle request failed")

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


def _maybe_import_climate_engine():  # lazy import avoids cycles
    from modules.climate import engine as climate_engine_mod

    return climate_engine_mod


async def _warm_mandi_to_redis(conn, state_s: str, district_s: str, settings) -> None:
    # Skip live OGD if cron/startup already warmed this district.
    existing = await redis_client.json_get_maybe(conn, ck.mandi_key(state_s, district_s))
    if isinstance(existing, dict) and existing.get("records"):
        return
    if not settings.ogd_api_key.strip():
        return
    try:
        from modules.market import ogd_client as ogd_client_mod

        rows = await ogd_client_mod.fetch_mandi_prices(
            state_s or None,
            district_s,
            None,
            settings.ogd_api_key.strip(),
            timeout=float(settings.market_tool_timeout_seconds),
        )
        key = ck.mandi_key(state_s, district_s)
        import time as _time

        await redis_client.json_setex(
            conn,
            key,
            ck.ttl_mandi(settings),
            {
                "records": rows,
                "saved_at": int(_time.time()),
                "district": district_s,
                "state": state_s or "",
            },
        )
    except Exception as e:
        logger.warning("Mandi hydrate on bundle failed: %s", e)


@router.post("/sync/push")
async def push_to_supabase() -> Dict[str, Any]:
    """Push unsynced SQLite rows to Supabase (direct or via QStash)."""
    settings = get_settings()
    if not settings.supabase_db_configured:
        return {"ok": False, "skipped": True, "reason": "Supabase DB not configured"}
    if settings.qstash_configured:
        from cache import qstash_client

        mid = await qstash_client.enqueue_sync_push(settings)
        return {"ok": True, "queued": True, "qstash_message_id": mid}

    from offline.supabase_sync import SupabaseSync

    try:
        return await SupabaseSync(settings).run_without_vectors()
    except Exception:
        logger.exception("SupabaseSync failed")
        raise
