"""Mandi price sync from data.gov.in into local SQLite."""

from __future__ import annotations

import logging
from typing import Any, Dict

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from config.settings import get_settings
from db.sqlite_client import get_mandi_prices_for_location, upsert_mandi_prices_bulk
from modules.market import ogd_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["market"])


def _format_sync_error(exc: BaseException) -> str:
    """Always non-empty; safe for client (no API keys)."""
    name = type(exc).__name__
    msg = str(exc).strip() or repr(exc)
    parts = [f"{name}: {msg}"]
    if isinstance(exc, httpx.HTTPStatusError):
        parts.append(ogd_client.format_ogd_http_error(exc))
    elif isinstance(exc, httpx.RequestError):
        parts.append(
            "Network error reaching api.data.gov.in (timeout, DNS, TLS, or firewall)."
        )
    return " | ".join(parts)


class MandiSyncBody(BaseModel):
    state: str = Field(..., min_length=1)
    district: str = Field(..., min_length=1)


@router.post("/market/sync")
async def sync_mandi_prices(body: MandiSyncBody) -> Dict[str, Any]:
    """Pull latest mandi rows for ``state`` + ``district`` from OGD into local SQLite."""
    settings = get_settings()
    key = (settings.ogd_api_key or "").strip()
    if not key:
        raise HTTPException(
            status_code=501,
            detail="OGD_API_KEY is not configured; cannot sync mandi prices server-side.",
        )
    state = body.state.strip()
    district = body.district.strip()
    try:
        records = await ogd_client.fetch_mandi_prices(
            state,
            district,
            None,
            key,
            timeout=float(settings.market_tool_timeout_seconds),
        )
    except Exception as e:
        logger.exception("Mandi sync OGD request failed")
        raise HTTPException(
            status_code=502,
            detail=f"OGD mandi fetch failed: {_format_sync_error(e)}",
        ) from e

    if not records:
        return {
            "ok": True,
            "state": state,
            "district": district,
            "synced_records": 0,
            "expires_in_seconds": settings.mandi_price_ttl_seconds,
            "message": "No records returned for this state/district.",
        }

    await upsert_mandi_prices_bulk(
        state,
        district,
        records,
        settings.mandi_price_ttl_seconds,
        settings,
    )
    return {
        "ok": True,
        "state": state,
        "district": district,
        "synced_records": len(records),
        "expires_in_seconds": settings.mandi_price_ttl_seconds,
    }


@router.get("/market/synced")
async def get_synced_mandi_prices(
    state: str = Query(..., min_length=1),
    district: str = Query(..., min_length=1),
    commodity: str = Query("", description="Optional commodity filter (e.g. Wheat)."),
) -> Dict[str, Any]:
    """Read locally synced mandi rows from SQLite."""
    settings = get_settings()
    comm = (commodity or "").strip() or None
    rows = await get_mandi_prices_for_location(
        state.strip(),
        district.strip(),
        settings,
        commodity=comm,
    )
    return {
        "ok": True,
        "state": state.strip(),
        "district": district.strip(),
        "commodity": comm,
        "count": len(rows),
        "records": rows,
    }
