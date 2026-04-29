"""Route farmer/query persistence between SQLite (offline + cache) and Supabase (online)."""

from __future__ import annotations

import logging
from typing import Optional

from config.settings import Settings, get_settings
from db import supabase_client
from db.sqlite_client import (
    get_farmer_twin,
    log_query,
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


async def persist_log_query(
    farmer_id: str,
    query_text: str,
    intent: str,
    response: str,
    data_source: str,
    connectivity: str,
    settings: Optional[Settings] = None,
) -> None:
    settings = settings or get_settings()
    offline = is_offline_context(connectivity)
    if offline:
        await log_query(farmer_id, query_text, intent, response, data_source, settings, synced=0)
        return
    if settings.supabase_db_configured:
        try:
            await supabase_client.insert_query_history_remote(
                farmer_id,
                query_text,
                intent,
                response,
                data_source,
                sqlite_timestamp_unix=None,
                settings=settings,
            )
            await log_query(farmer_id, query_text, intent, response, data_source, settings, synced=1)
        except Exception as e:
            logger.warning("Remote query log failed, queuing for sync: %s", e)
            await log_query(farmer_id, query_text, intent, response, data_source, settings, synced=0)
    else:
        await log_query(farmer_id, query_text, intent, response, data_source, settings, synced=1)
