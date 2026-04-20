"""Optional sync job: placeholder for downloading public mandi CSV."""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from config.settings import get_settings
from db.sqlite_client import set_sync_meta

logger = logging.getLogger(__name__)


class SyncJob:
    """Download fresh data when online (stub — validates HTTP + records timestamp)."""

    def __init__(self, url: Optional[str] = None):
        self.url = url or "https://api.open-meteo.com/v1/forecast?latitude=20&longitude=78&current_weather=true"

    async def run(self) -> dict:
        settings = get_settings()
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                r = await client.get(self.url)
                r.raise_for_status()
        except Exception as e:
            logger.warning("Sync probe failed: %s", e)
            return {"ok": False, "error": str(e)}
        await set_sync_meta("last_sync_probe", "ok", settings)
        return {"ok": True, "status": r.status_code}
