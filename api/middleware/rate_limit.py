"""SQLite-backed rate limiting."""

from __future__ import annotations

from fastapi import HTTPException, Request

from config.settings import get_settings
from db.sqlite_client import check_rate_limit


async def enforce_rate_limit(request: Request) -> None:
    farmer_id = request.headers.get("X-Farmer-Id") or request.query_params.get("farmer_id")
    if not farmer_id:
        # For POST body we cannot read easily here; routes pass farmer_id in body — use optional
        return
    settings = get_settings()
    ok, err = await check_rate_limit(farmer_id, settings)
    if not ok:
        raise HTTPException(status_code=429, detail=err or "Too many requests")
