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

CREATE TABLE IF NOT EXISTS conversation_metadata (
    conversation_id TEXT PRIMARY KEY,
    farmer_id TEXT NOT NULL,
    title TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    synced INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS query_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
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

CREATE TABLE IF NOT EXISTS pending_deletes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    farmer_id TEXT NOT NULL,
    queued_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS pending_deletes_type_idx ON pending_deletes(entity_type);
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

    cur = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='conversation_metadata'"
    )
    cm_row = await cur.fetchone()
    if not cm_row:
        await db.executescript(
            """
            CREATE TABLE conversation_metadata (
                conversation_id TEXT PRIMARY KEY,
                farmer_id TEXT NOT NULL,
                title TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                synced INTEGER NOT NULL DEFAULT 0
            );
            """
        )

    cm = await _cols("conversation_metadata")
    if cm and "synced" not in cm:
        await db.execute(
            "ALTER TABLE conversation_metadata ADD COLUMN synced INTEGER NOT NULL DEFAULT 0"
        )

    cur = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='pending_deletes'"
    )
    if not await cur.fetchone():
        await db.executescript(
            """
            CREATE TABLE pending_deletes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                farmer_id TEXT NOT NULL,
                queued_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS pending_deletes_type_idx ON pending_deletes(entity_type);
            """
        )



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


async def upsert_conversation_metadata(
    conversation_id: str,
    farmer_id: str,
    title: Optional[str],
    settings: Optional[Settings] = None,
    *,
    synced: int = 0,
) -> None:
    now = int(time.time())
    if not conversation_id.strip():
        raise ValueError("conversation_id is required")
    async with get_connection(settings) as db:
        await db.execute(
            """
            INSERT INTO conversation_metadata (
                conversation_id, farmer_id, title, created_at, updated_at, synced
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
              farmer_id = excluded.farmer_id,
              title = COALESCE(excluded.title, conversation_metadata.title),
              updated_at = excluded.updated_at,
              synced = excluded.synced
            """,
            (
                conversation_id.strip(),
                farmer_id,
                title,
                now,
                now,
                synced,
            ),
        )
        await db.commit()


async def get_conversation_metadata(
    conversation_id: str, settings: Optional[Settings] = None
) -> Optional[Dict[str, Any]]:
    if not conversation_id.strip():
        return None
    async with get_connection(settings) as db:
        cur = await db.execute(
            """
            SELECT conversation_id, farmer_id, title, created_at, updated_at, synced
            FROM conversation_metadata WHERE conversation_id = ?
            """,
            (conversation_id.strip(),),
        )
        row = await cur.fetchone()
    return dict(row) if row else None


async def get_conversations_by_farmer(
    farmer_id: str, settings: Optional[Settings] = None
) -> List[Dict[str, Any]]:
    async with get_connection(settings) as db:
        cur = await db.execute(
            """
            SELECT conversation_id, farmer_id, title, created_at, updated_at
            FROM conversation_metadata
            WHERE farmer_id = ?
            ORDER BY created_at DESC
            """,
            (farmer_id,),
        )
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def delete_conversation_local(
    conversation_id: str, settings: Optional[Settings] = None
) -> None:
    """Remove all query_history turns and conversation_metadata for one session."""
    cid = (conversation_id or "").strip()
    if not cid:
        return
    async with get_connection(settings) as db:
        await db.execute("BEGIN IMMEDIATE")
        try:
            await db.execute("DELETE FROM query_history WHERE conversation_id = ?", (cid,))
            await db.execute(
                "DELETE FROM conversation_metadata WHERE conversation_id = ?", (cid,)
            )
        except Exception:
            await db.rollback()
            raise
        await db.commit()


async def queue_pending_delete(
    entity_type: str,
    entity_id: str,
    farmer_id: str,
    settings: Optional[Settings] = None,
) -> None:
    """Record a deletion to propagate to Supabase when back online."""
    now = int(time.time())
    eid = (entity_id or "").strip()
    if not eid or not (entity_type or "").strip():
        return
    async with get_connection(settings) as db:
        await db.execute(
            """
            INSERT INTO pending_deletes (entity_type, entity_id, farmer_id, queued_at)
            VALUES (?, ?, ?, ?)
            """,
            (entity_type.strip(), eid, farmer_id, now),
        )
        await db.commit()


async def fetch_pending_deletes(
    entity_type: str, settings: Optional[Settings] = None
) -> List[Dict[str, Any]]:
    et = (entity_type or "").strip()
    if not et:
        return []
    async with get_connection(settings) as db:
        cur = await db.execute(
            """
            SELECT id, entity_type, entity_id, farmer_id, queued_at
            FROM pending_deletes
            WHERE entity_type = ?
            ORDER BY queued_at ASC, id ASC
            """,
            (et,),
        )
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def remove_pending_delete(
    row_id: int, settings: Optional[Settings] = None
) -> None:
    async with get_connection(settings) as db:
        await db.execute("DELETE FROM pending_deletes WHERE id = ?", (row_id,))
        await db.commit()


async def log_query(
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
                query_text, intent, response, timestamp, data_source, conversation_id, synced
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (query_text, intent, response, now, data_source, conversation_id, synced),
        )
        await db.commit()


async def get_query_history_for_conversation(
    conversation_id: str, settings: Optional[Settings] = None
) -> List[Dict[str, Any]]:
    """All logged turns for a session, oldest first."""
    cid = (conversation_id or "").strip()
    if not cid:
        return []
    async with get_connection(settings) as db:
        cur = await db.execute(
            """
            SELECT id, query_text, intent, response, timestamp, data_source, conversation_id
            FROM query_history
            WHERE conversation_id = ?
            ORDER BY timestamp ASC
            """,
            (cid,),
        )
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def get_last_n_turns(
    conversation_id: str,
    n: int = 3,
    settings: Optional[Settings] = None,
) -> List[Dict[str, Any]]:
    """Return last n turns oldest-first for a conversation (prior exchanges only)."""
    cid = (conversation_id or "").strip()
    if not cid:
        return []
    async with get_connection(settings) as db:
        cur = await db.execute(
            """
            SELECT query_text, response FROM query_history
            WHERE conversation_id = ?
            ORDER BY timestamp DESC LIMIT ?
            """,
            (cid, n),
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


async def get_weather_cache(
    location_key: str, settings: Optional[Settings] = None
) -> Optional[Tuple[Dict[str, Any], int, int]]:
    """Return (payload dict, fetched_at unix, expires_at unix) if row exists and not expired."""
    now = int(time.time())
    async with get_connection(settings) as db:
        cur = await db.execute(
            """
            SELECT payload, fetched_at, expires_at FROM weather_cache
            WHERE location_key = ? AND expires_at > ?
            """,
            (location_key, now),
        )
        row = await cur.fetchone()
        if not row:
            return None
        return (json.loads(row["payload"]), int(row["fetched_at"]), int(row["expires_at"]))


async def set_weather_cache(
    location_key: str,
    payload: Dict[str, Any],
    ttl_seconds: int,
    settings: Optional[Settings] = None,
) -> Tuple[int, int]:
    """Upsert cached weather payload; returns (fetched_at, expires_at) unix timestamps."""
    now = int(time.time())
    expires = now + max(60, ttl_seconds)
    blob = json.dumps(payload, ensure_ascii=False)
    async with get_connection(settings) as db:
        await db.execute(
            """
            INSERT INTO weather_cache (location_key, fetched_at, expires_at, payload)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(location_key) DO UPDATE SET
              fetched_at = excluded.fetched_at,
              expires_at = excluded.expires_at,
              payload = excluded.payload
            """,
            (location_key, now, expires, blob),
        )
        await db.commit()
    return (now, expires)


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
            SELECT id, query_text, intent, response, timestamp, data_source, conversation_id
            FROM query_history WHERE synced = 0 ORDER BY id
            """
        )
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def fetch_unsynced_conversation_metadata(
    settings: Optional[Settings] = None,
) -> List[Dict[str, Any]]:
    async with get_connection(settings) as db:
        cur = await db.execute(
            """
            SELECT conversation_id, farmer_id, title, created_at, updated_at
            FROM conversation_metadata WHERE synced = 0 ORDER BY created_at
            """
        )
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def mark_conversation_metadata_synced(
    conversation_id: str, settings: Optional[Settings] = None
) -> None:
    async with get_connection(settings) as db:
        await db.execute(
            "UPDATE conversation_metadata SET synced = 1 WHERE conversation_id = ?",
            (conversation_id,),
        )
        await db.commit()


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
