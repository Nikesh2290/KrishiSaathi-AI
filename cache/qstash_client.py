"""Publish QStash messages to PUBLIC_API_BASE_URL job endpoints."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


def _callbacks_base(settings: Settings) -> str:
    base = settings.public_api_base_url.strip().rstrip("/")
    if not base:
        raise RuntimeError("PUBLIC_API_BASE_URL is required for QStash")
    return base


async def enqueue_persist_query(
    settings: Settings,
    body: Dict[str, Any],
    *,
    dedupe_id: Optional[str] = None,
) -> Optional[str]:
    if not settings.qstash_configured:
        return None

    url = f"{_callbacks_base(settings)}/api/v1/internal/qstash/persist-query"
    return await _publish(settings, url, body, dedupe_id=dedupe_id)


async def enqueue_sync_push(
    settings: Settings,
    *,
    dedupe_id: Optional[str] = None,
) -> Optional[str]:
    if not settings.qstash_configured:
        return None
    url = f"{_callbacks_base(settings)}/api/v1/internal/qstash/sync-push"
    return await _publish(settings, url, {}, dedupe_id=dedupe_id)


async def enqueue_regenerate_bundle(
    settings: Settings,
    bundle_args: Dict[str, Any],
    *,
    dedupe_id: Optional[str] = None,
) -> Optional[str]:
    if not settings.qstash_configured:
        return None
    url = f"{_callbacks_base(settings)}/api/v1/internal/qstash/regenerate-bundle"
    return await _publish(settings, url, bundle_args, dedupe_id=dedupe_id)


async def enqueue_escalate_llm(settings: Settings, body: Dict[str, Any]) -> Optional[str]:
    """Queue optional heavy-model follow-up logging (minimal processing in worker)."""
    if not settings.qstash_configured:
        return None
    url = f"{_callbacks_base(settings)}/api/v1/internal/qstash/escalate-llm"
    return await _publish(settings, url, body, dedupe_id=None)


async def _publish(
    settings: Settings,
    destination: str,
    body: Dict[str, Any],
    *,
    dedupe_id: Optional[str],
) -> str:
    from qstash import AsyncQStash

    token = settings.qstash_token.strip()
    qs = AsyncQStash(token, base_url=(settings.qstash_url.strip() or None))
    kw: Dict[str, Any] = {
        "url": destination,
        "body": json.dumps(body),
        "content_type": "application/json",
        "headers": {"Content-Type": "application/json"},
    }
    if dedupe_id:
        kw["deduplication_id"] = dedupe_id
        kw["content_based_deduplication"] = False

    res = await qs.message.publish(**kw)
    return str(getattr(res, "message_id", res))
async def safe_enqueue_persist(
    settings: Optional[Settings],
    payload: Dict[str, Any],
    *,
    sqlite_row_id: int,
) -> None:
    s = settings or get_settings()
    try:
        await enqueue_persist_query(s, payload, dedupe_id=f"persist-query-{sqlite_row_id}")
    except Exception as e:
        logger.warning("QStash enqueue persist-query failed (will retry via SQLite sync later): %s", e)
