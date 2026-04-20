"""SQLite persistence: farmer twin, caches, query history, rate limits."""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

import aiosqlite

from config.settings import Settings, get_settings
from models.farmer import FarmerTwin


SCHEMA = """
CREATE TABLE IF NOT EXISTS farmer_twin (
    farmer_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS weather_cache (
    location_key TEXT PRIMARY KEY,
    fetched_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS price_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    crop TEXT NOT NULL,
    mandi TEXT NOT NULL,
    fetched_at INTEGER NOT NULL,
    price_inr REAL NOT NULL,
    unit TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS query_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    farmer_id TEXT,
    query_text TEXT,
    intent TEXT,
    response TEXT,
    timestamp INTEGER NOT NULL,
    data_source TEXT
);

CREATE TABLE IF NOT EXISTS rate_limit (
    farmer_id TEXT PRIMARY KEY,
    window_start INTEGER NOT NULL,
    count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sync_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);
"""


async def init_db(settings: Optional[Settings] = None) -> None:
    settings = settings or get_settings()
    db_path = os.path.abspath(settings.database_path)
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    async with aiosqlite.connect(settings.database_path) as db:
        await db.executescript(SCHEMA)
        await db.commit()


@asynccontextmanager
async def get_connection(settings: Optional[Settings] = None) -> AsyncIterator[aiosqlite.Connection]:
    settings = settings or get_settings()
    db = await aiosqlite.connect(settings.database_path)
    db.row_factory = aiosqlite.Row
    try:
        yield db
    finally:
        await db.close()


async def get_farmer_twin(farmer_id: str, settings: Optional[Settings] = None) -> Optional[FarmerTwin]:
    async with get_connection(settings) as db:
        cur = await db.execute(
            "SELECT payload FROM farmer_twin WHERE farmer_id = ?", (farmer_id,)
        )
        row = await cur.fetchone()
        if not row:
            return None
        data = json.loads(row["payload"])
        return FarmerTwin.model_validate(data)


async def upsert_farmer_twin(twin: FarmerTwin, settings: Optional[Settings] = None) -> None:
    now = int(time.time())
    payload = twin.model_dump_json()
    async with get_connection(settings) as db:
        await db.execute(
            """
            INSERT INTO farmer_twin (farmer_id, payload, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(farmer_id) DO UPDATE SET
              payload = excluded.payload,
              updated_at = excluded.updated_at
            """,
            (twin.farmer_id, payload, now),
        )
        await db.commit()


async def log_query(
    farmer_id: str,
    query_text: str,
    intent: str,
    response: str,
    data_source: str,
    settings: Optional[Settings] = None,
) -> None:
    now = int(time.time())
    async with get_connection(settings) as db:
        await db.execute(
            """
            INSERT INTO query_history (farmer_id, query_text, intent, response, timestamp, data_source)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (farmer_id, query_text, intent, response, now, data_source),
        )
        await db.commit()


async def check_rate_limit(
    farmer_id: str, settings: Optional[Settings] = None
) -> tuple[bool, Optional[str]]:
    """Return (allowed, error_message)."""
    settings = settings or get_settings()
    limit = settings.rate_limit_per_minute
    now = int(time.time())
    window = 60
    window_start = now - (now % window)

    async with get_connection(settings) as db:
        cur = await db.execute(
            "SELECT window_start, count FROM rate_limit WHERE farmer_id = ?",
            (farmer_id,),
        )
        row = await cur.fetchone()
        if not row:
            await db.execute(
                "INSERT INTO rate_limit (farmer_id, window_start, count) VALUES (?, ?, ?)",
                (farmer_id, window_start, 1),
            )
            await db.commit()
            return True, None

        ws, cnt = row["window_start"], row["count"]
        if ws != window_start:
            await db.execute(
                "UPDATE rate_limit SET window_start = ?, count = 1 WHERE farmer_id = ?",
                (window_start, farmer_id),
            )
            await db.commit()
            return True, None

        if cnt >= limit:
            return False, f"Rate limit exceeded: {limit} requests per minute."

        await db.execute(
            "UPDATE rate_limit SET count = count + 1 WHERE farmer_id = ?",
            (farmer_id,),
        )
        await db.commit()
        return True, None


async def get_sync_meta(key: str, settings: Optional[Settings] = None) -> Optional[str]:
    async with get_connection(settings) as db:
        cur = await db.execute("SELECT value FROM sync_meta WHERE key = ?", (key,))
        row = await cur.fetchone()
        return row["value"] if row else None


async def set_sync_meta(key: str, value: str, settings: Optional[Settings] = None) -> None:
    now = int(time.time())
    async with get_connection(settings) as db:
        await db.execute(
            """
            INSERT INTO sync_meta (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (key, value, now),
        )
        await db.commit()
