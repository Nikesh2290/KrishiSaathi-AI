"""Route farmer/query persistence between SQLite (offline + cache) and Supabase (online)."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from config.settings import Settings, get_settings
from db import supabase_client
from db.sqlite_client import (
    delete_conversation_local,
    get_conversation_metadata,
    get_conversations_by_farmer,
    get_farmer_twin,
    get_query_history_for_conversation,
    log_query,
    queue_pending_delete,
    upsert_conversation_metadata,
    upsert_farmer_twin,
)
from models.farmer import FarmerTwin

from cache import qstash_client
from cache.context_builder import append_turn_update_redis

logger = logging.getLogger(__name__)


def is_offline_context(connectivity: str) -> bool:
    return (connectivity or "").lower() == "offline"


async def resolve_farmer_twin(
    farmer_id: str,
    connectivity: str,
    settings: Optional[Settings] = None,
) -> Optional[FarmerTwin]:
    """Agent graph: prefer Supabase when online and configured, else SQLite."""
    settings = settings or get_settings()
    if is_offline_context(connectivity) or not settings.supabase_db_configured:
        return await get_farmer_twin(farmer_id, settings)
    remote = await supabase_client.get_farmer_twin_remote(farmer_id, settings)
    if remote is not None:
        return remote
    return await get_farmer_twin(farmer_id, settings)


async def persist_farmer_twin(
    twin: FarmerTwin,
    connectivity: str,
    settings: Optional[Settings] = None,
) -> None:
    settings = settings or get_settings()
    offline = is_offline_context(connectivity)
    if offline:
        await upsert_farmer_twin(twin, settings, synced=0)
        return
    if settings.supabase_db_configured:
        try:
            await supabase_client.upsert_farmer_twin_remote(twin, settings)
            await upsert_farmer_twin(twin, settings, synced=1)
        except Exception as e:
            logger.warning("Remote farmer upsert failed, caching locally: %s", e)
            await upsert_farmer_twin(twin, settings, synced=0)
    else:
        await upsert_farmer_twin(twin, settings, synced=1)

    if settings.redis_configured:
        try:
            from cache.redis_client import redis_from_settings
            from cache.context_builder import cache_farmer_twin

            r = redis_from_settings(settings)
            await cache_farmer_twin(r, twin, settings)
        except Exception as e:
            logger.debug("Redis twin cache after persist skipped: %s", e)


async def persist_conversation_metadata(
    conversation_id: str,
    farmer_id: str,
    title: Optional[str],
    connectivity: str,
    settings: Optional[Settings] = None,
) -> None:
    """Upsert thread metadata to SQLite and/or Supabase."""
    settings = settings or get_settings()
    offline = is_offline_context(connectivity)
    if offline:
        await upsert_conversation_metadata(
            conversation_id, farmer_id, title, settings, synced=0
        )
        return
    if settings.supabase_db_configured:
        try:
            await supabase_client.upsert_conversation_metadata_remote(
                conversation_id, farmer_id, title, settings=settings
            )
            await upsert_conversation_metadata(
                conversation_id, farmer_id, title, settings, synced=1
            )
        except Exception as e:
            logger.warning("Remote conversation_metadata upsert failed, queuing locally: %s", e)
            await upsert_conversation_metadata(
                conversation_id, farmer_id, title, settings, synced=0
            )
    else:
        await upsert_conversation_metadata(
            conversation_id, farmer_id, title, settings, synced=1
        )


async def resolve_conversations_by_farmer(
    farmer_id: str,
    connectivity: str,
    settings: Optional[Settings] = None,
) -> List[Dict[str, Any]]:
    """List conversation threads for a farmer (Supabase when online, else SQLite)."""
    settings = settings or get_settings()
    if is_offline_context(connectivity) or not settings.supabase_db_configured:
        return await get_conversations_by_farmer(farmer_id, settings)
    try:
        rows = await supabase_client.get_conversations_by_farmer_remote(farmer_id, settings)
        if rows:
            return rows
    except Exception as e:
        logger.warning("Remote conversation list failed, using local cache: %s", e)
    return await get_conversations_by_farmer(farmer_id, settings)


async def resolve_conversation_history(
    farmer_id: str,
    conversation_id: str,
    connectivity: str,
    settings: Optional[Settings] = None,
) -> Optional[Dict[str, Any]]:
    """Session metadata plus query/response turns when the thread exists and belongs to farmer_id."""
    settings = settings or get_settings()
    cid = (conversation_id or "").strip()
    if not cid:
        return None

    offline = is_offline_context(connectivity) or not settings.supabase_db_configured

    if not offline:
        try:
            meta = await supabase_client.get_conversation_metadata_remote(cid, settings)
            if meta is not None and str(meta.get("farmer_id")) == str(farmer_id):
                try:
                    messages = await supabase_client.get_query_history_by_conversation_remote(
                        cid, settings
                    )
                except Exception as e:
                    logger.warning(
                        "Remote query history failed, using local cache: %s", e
                    )
                    messages = await get_query_history_for_conversation(cid, settings)
                return {"meta": meta, "messages": messages}
            if meta is not None:
                return None
        except Exception as e:
            logger.warning("Remote conversation history path failed, using local: %s", e)

    meta = await get_conversation_metadata(cid, settings)
    if not meta or str(meta.get("farmer_id")) != str(farmer_id):
        return None
    messages = await get_query_history_for_conversation(cid, settings)
    return {"meta": meta, "messages": messages}


async def delete_conversation(
    farmer_id: str,
    conversation_id: str,
    connectivity: str,
    settings: Optional[Settings] = None,
) -> Optional[bool]:
    """Verify ownership, delete local data and remote when online; queue for Supabase if offline or remote fails."""
    settings = settings or get_settings()
    cid = (conversation_id or "").strip()
    if not cid:
        return None

    offline = is_offline_context(connectivity) or not settings.supabase_db_configured

    owner_verified_via_remote = False
    if not offline:
        try:
            meta = await supabase_client.get_conversation_metadata_remote(cid, settings)
            if meta is not None and str(meta.get("farmer_id")) == str(farmer_id):
                owner_verified_via_remote = True
            elif meta is not None:
                return None
        except Exception as e:
            logger.warning("Remote conversation metadata for delete failed, using local: %s", e)

    if not owner_verified_via_remote:
        meta = await get_conversation_metadata(cid, settings)
        if not meta or str(meta.get("farmer_id")) != str(farmer_id):
            return None

    remote_ok = False
    if not offline:
        try:
            await supabase_client.delete_conversation_remote(cid, settings)
            remote_ok = True
        except Exception as e:
            logger.warning("Remote conversation delete failed: %s", e)

    await delete_conversation_local(cid, settings)

    need_queue = bool(settings.supabase_db_configured) and (offline or not remote_ok)
    if need_queue:
        await queue_pending_delete("conversation", cid, str(farmer_id), settings)

    return True


async def persist_log_query(
    query_text: str,
    intent: str,
    response: str,
    data_source: str,
    connectivity: str,
    *,
    farmer_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> None:
    """Log a query/response turn. Ensures conversation_metadata exists when farmer_id is known.

    Online + Supabase + QStash: writes SQLite ``synced=0`` immediately, enqueues remote insert.

    Offline: SQLite ``synced=0`` only.
    """
    settings = settings or get_settings()
    cid = (conversation_id or "").strip() or None
    if cid and farmer_id:
        try:
            await persist_conversation_metadata(
                cid, farmer_id, title=None, connectivity=connectivity, settings=settings
            )
        except Exception as e:
            logger.warning("conversation metadata touch failed: %s", e)

    offline = is_offline_context(connectivity)

    fid = (farmer_id or "").strip()

    async def _after_append() -> None:
        if fid and cid and settings.redis_configured:
            try:
                await append_turn_update_redis(
                    fid, cid, query_text, response[:3500], settings=settings
                )
            except Exception as e:
                logger.debug("redis session append skipped: %s", e)

    if offline:
        row_id = await log_query(
            query_text,
            intent,
            response,
            data_source,
            settings,
            synced=0,
            conversation_id=cid,
        )
        logger.debug("log_query sqlite id=%s (offline)", row_id)
        await _after_append()
        return

    if settings.supabase_db_configured and settings.qstash_configured:
        row_id = await log_query(
            query_text,
            intent,
            response,
            data_source,
            settings,
            synced=0,
            conversation_id=cid,
        )
        ts = int(time.time())
        payload = {
            "sqlite_row_id": row_id,
            "query_text": query_text,
            "intent": intent,
            "response": response,
            "data_source": data_source,
            "conversation_id": cid,
            "sqlite_timestamp_unix": ts,
            "farmer_id": fid,
        }
        await qstash_client.safe_enqueue_persist(settings, payload, sqlite_row_id=row_id)
        await _after_append()
        return

    if settings.supabase_db_configured:
        try:
            await supabase_client.insert_query_history_remote(
                query_text,
                intent,
                response,
                data_source,
                sqlite_timestamp_unix=None,
                conversation_id=cid,
                settings=settings,
            )
            await log_query(
                query_text,
                intent,
                response,
                data_source,
                settings,
                synced=1,
                conversation_id=cid,
            )
        except Exception as e:
            logger.warning("Remote query log failed, queuing for sync: %s", e)
            await log_query(
                query_text,
                intent,
                response,
                data_source,
                settings,
                synced=0,
                conversation_id=cid,
            )
    else:
        await log_query(
            query_text,
            intent,
            response,
            data_source,
            settings,
            synced=1,
            conversation_id=cid,
        )
    await _after_append()
