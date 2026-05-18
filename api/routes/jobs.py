"""QStash webhook receivers (PUBLIC_API_BASE_URL must match signed URL)."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request, Response

from config.settings import get_settings
from db import sqlite_client
from db import supabase_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/internal/qstash", tags=["internal-qstash"])


async def _verify_qstash(request: Request) -> Dict[str, Any]:
    raw = await request.body()
    body_text = raw.decode("utf-8")
    sig = request.headers.get("Upstash-Signature") or ""

    settings = get_settings()
    try:
        from qstash import Receiver

        Receiver(
            settings.qstash_current_signing_key,
            settings.qstash_next_signing_key,
        ).verify(signature=sig, body=body_text)
    except Exception as e:
        logger.warning("QStash verification failed: %s", e)
        raise HTTPException(status_code=401, detail="invalid signature") from e

    try:
        return json.loads(body_text) if body_text.strip() else {}
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail="bad json body") from e


@router.post("/persist-query")
async def qstash_persist_query(request: Request) -> Dict[str, Any]:
    payload = await _verify_qstash(request)

    sqlite_row_id = int(payload.get("sqlite_row_id"))
    ts = payload.get("sqlite_timestamp_unix")
    ts_i = int(ts) if ts is not None else None
    conversation_id = payload.get("conversation_id")

    settings = get_settings()

    # Conversation metadata may not yet be synced to Supabase (async sync).
    # Upsert a stub row first so the FK constraint on query_history is satisfied.
    if conversation_id:
        farmer_id = str(payload.get("farmer_id") or "")
        if farmer_id:
            try:
                await supabase_client.upsert_conversation_metadata_remote(
                    conversation_id, farmer_id, None, settings=settings
                )
            except Exception:
                logger.warning("persist-query: stub conversation upsert failed", exc_info=True)

    await supabase_client.insert_query_history_remote(
        payload.get("query_text") or "",
        payload.get("intent") or "",
        payload.get("response") or "",
        payload.get("data_source") or "live",
        sqlite_timestamp_unix=ts_i,
        conversation_id=conversation_id,
        settings=settings,
    )
    await sqlite_client.mark_query_history_synced(sqlite_row_id, settings)
    return {"ok": True}


@router.post("/sync-push")
async def qstash_sync_push(request: Request) -> Dict[str, Any]:
    await _verify_qstash(request)
    settings = get_settings()
    if not settings.supabase_db_configured:
        raise HTTPException(status_code=503, detail="supabase_not_configured")
    from offline.supabase_sync import SupabaseSync

    return await SupabaseSync(settings).run_without_vectors()


@router.post("/regenerate-bundle")
async def qstash_regenerate_bundle(request: Request) -> Dict[str, Any]:
    body = await _verify_qstash(request)
    settings = get_settings()

    try:
        from api.routes.sync import invalidate_bundle_cache_all

        invalidate_bundle_cache_all()
    except Exception:
        logger.exception("bundle cache invalidate failed")

    state_s = body.get("state") or ""
    district_s = body.get("district") or ""
    if isinstance(state_s, str) and state_s.strip() and isinstance(district_s, str) and district_s.strip():
        try:
            from offline.bundle_builder import build_gzip_bundle

            _raw, ver = build_gzip_bundle(state_s.strip(), district_s.strip())
            if settings.redis_configured:
                from cache.redis_client import redis_from_settings, json_setex

                rk = (
                    "sync_bundle_meta:"
                    f"{state_s.strip().lower()}:{district_s.strip().lower()}:{int(time.time())}"
                )
                r = redis_from_settings(settings)
                await json_setex(
                    r,
                    rk,
                    int(settings.sync_bundle_cache_ttl_seconds),
                    {"version": ver},
                )
        except Exception:
            logger.exception("optional bundle regeneration step failed")

    return {"ok": True}


@router.post("/escalate-llm")
async def qstash_escalate_llm(request: Request) -> Response:
    payload = await _verify_qstash(request)
    cid = payload.get("conversation_id")
    fid = payload.get("farmer_id")
    hint = payload.get("hint")

    logger.info(
        "escalate-llm queued event conversation_id=%s farmer_id=%s hint=%s",
        cid,
        fid,
        hint,
    )

    settings = get_settings()
    if settings.redis_configured and cid:
        try:
            from cache import cache_keys
            from cache.redis_client import redis_from_settings, json_setex

            rk = cache_keys.warmup_meta_key(f"escalate_ev:{cid}:{int(time.time())}")
            r = redis_from_settings(settings)
            await json_setex(r, rk, 3600, payload)
        except Exception:
            logger.exception("Redis escalation event store failed")

    return Response(content='{"ok":true}', media_type="application/json")
