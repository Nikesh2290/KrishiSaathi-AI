"""SQLite persistence: farmer twin, caches, query history."""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import aiosqlite

from config.settings import Settings, get_settings
from models.farmer import FarmerTwin


SCHEMA = """
CREATE TABLE IF NOT EXISTS farmer_twin (
    farmer_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    updated_at INTEGER NOT NULL,
    synced INTEGER NOT NULL DEFAULT 0
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
    data_source TEXT,
    conversation_id TEXT,
    synced INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sync_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS auth_state (
    user_id TEXT PRIMARY KEY,
    access_token TEXT,
    refresh_token TEXT,
    expires_at INTEGER
);
"""


async def _migrate_existing_db(db: aiosqlite.Connection) -> None:
    """Upgrade older DB files (ADD COLUMN) when tables already existed without new columns."""

    async def _cols(table: str) -> List[str]:
        cur = await db.execute(f"PRAGMA table_info({table})")
        rows = await cur.fetchall()
        return [r[1] for r in rows]

    ft = await _cols("farmer_twin")
    if ft and "synced" not in ft:
        await db.execute("ALTER TABLE farmer_twin ADD COLUMN synced INTEGER NOT NULL DEFAULT 0")

    qh = await _cols("query_history")
    if qh and "synced" not in qh:
        await db.execute("ALTER TABLE query_history ADD COLUMN synced INTEGER NOT NULL DEFAULT 0")
    if qh and "conversation_id" not in qh:
        await db.execute("ALTER TABLE query_history ADD COLUMN conversation_id TEXT")


async def init_db(settings: Optional[Settings] = None) -> None:
    settings = settings or get_settings()
    db_path = os.path.abspath(settings.database_path)
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    async with aiosqlite.connect(settings.database_path) as db:
        await db.executescript(SCHEMA)
        await _migrate_existing_db(db)
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


async def upsert_farmer_twin(
    twin: FarmerTwin,
    settings: Optional[Settings] = None,
    *,
    synced: int = 1,
) -> None:
    now = int(time.time())
    payload = twin.model_dump_json()
    async with get_connection(settings) as db:
        await db.execute(
            """
            INSERT INTO farmer_twin (farmer_id, payload, updated_at, synced)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(farmer_id) DO UPDATE SET
              payload = excluded.payload,
              updated_at = excluded.updated_at,
              synced = excluded.synced
            """,
            (twin.farmer_id, payload, now, synced),
        )
        await db.commit()


async def log_query(
    farmer_id: str,
    query_text: str,
    intent: str,
    response: str,
    data_source: str,
    settings: Optional[Settings] = None,
    *,
    synced: int = 0,
    conversation_id: Optional[str] = None,
) -> None:
    now = int(time.time())
    async with get_connection(settings) as db:
        await db.execute(
            """
            INSERT INTO query_history (
                farmer_id, query_text, intent, response, timestamp, data_source, conversation_id, synced
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (farmer_id, query_text, intent, response, now, data_source, conversation_id, synced),
        )
        await db.commit()


async def get_last_n_turns(
    farmer_id: str,
    conversation_id: str,
    n: int = 3,
    settings: Optional[Settings] = None,
) -> List[Dict[str, Any]]:
    """Return last n turns oldest-first for a conversation (prior exchanges only)."""
    async with get_connection(settings) as db:
        cur = await db.execute(
            """
            SELECT query_text, response FROM query_history
            WHERE farmer_id = ? AND conversation_id = ?
            ORDER BY timestamp DESC LIMIT ?
            """,
            (farmer_id, conversation_id, n),
        )
        rows = await cur.fetchall()
    return [dict(r) for r in reversed(rows)]


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


# --- Supabase sync helpers ---


async def fetch_unsynced_farmer_twins(
    settings: Optional[Settings] = None,
) -> List[Tuple[str, FarmerTwin]]:
    async with get_connection(settings) as db:
        cur = await db.execute(
            "SELECT farmer_id, payload FROM farmer_twin WHERE synced = 0"
        )
        rows = await cur.fetchall()
    out: List[Tuple[str, FarmerTwin]] = []
    for r in rows:
        data = json.loads(r["payload"])
        out.append((r["farmer_id"], FarmerTwin.model_validate(data)))
    return out


async def mark_farmer_twin_synced(farmer_id: str, settings: Optional[Settings] = None) -> None:
    async with get_connection(settings) as db:
        await db.execute(
            "UPDATE farmer_twin SET synced = 1 WHERE farmer_id = ?", (farmer_id,)
        )
        await db.commit()


async def fetch_unsynced_query_rows(
    settings: Optional[Settings] = None,
) -> List[Dict[str, Any]]:
    async with get_connection(settings) as db:
        cur = await db.execute(
            """
            SELECT id, farmer_id, query_text, intent, response, timestamp, data_source, conversation_id
            FROM query_history WHERE synced = 0 ORDER BY id
            """
        )
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def mark_query_history_synced(row_id: int, settings: Optional[Settings] = None) -> None:
    async with get_connection(settings) as db:
        await db.execute("UPDATE query_history SET synced = 1 WHERE id = ?", (row_id,))
        await db.commit()


# --- Local auth cache (offline session recall) ---


async def upsert_auth_state(
    user_id: str,
    access_token: str,
    refresh_token: str,
    expires_at: int,
    settings: Optional[Settings] = None,
) -> None:
    async with get_connection(settings) as db:
        await db.execute(
            """
            INSERT INTO auth_state (user_id, access_token, refresh_token, expires_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
              access_token = excluded.access_token,
              refresh_token = excluded.refresh_token,
              expires_at = excluded.expires_at
            """,
            (user_id, access_token, refresh_token, expires_at),
        )
        await db.commit()


async def get_auth_state(
    user_id: str, settings: Optional[Settings] = None
) -> Optional[Dict[str, Any]]:
    async with get_connection(settings) as db:
        cur = await db.execute(
            "SELECT user_id, access_token, refresh_token, expires_at FROM auth_state WHERE user_id = ?",
            (user_id,),
        )
        row = await cur.fetchone()
        if not row:
            return None
        return dict(row)
