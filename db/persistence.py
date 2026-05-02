"""Route farmer/query persistence between SQLite (offline + cache) and Supabase (online)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from config.settings import Settings, get_settings
from db import supabase_client
from db.sqlite_client import (
    get_conversations_by_farmer,
    get_farmer_twin,
    log_query,
    upsert_conversation_metadata,
    upsert_farmer_twin,
)
from models.farmer import FarmerTwin

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
    """Log a query/response turn. Ensures conversation_metadata exists when farmer_id is known."""
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
    if offline:
        await log_query(
            query_text,
            intent,
            response,
            data_source,
            settings,
            synced=0,
            conversation_id=cid,
        )
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
