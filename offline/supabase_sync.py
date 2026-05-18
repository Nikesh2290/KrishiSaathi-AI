"""Push offline SQLite data and scheme embeddings to Supabase (Postgres + pgvector)."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from config.settings import Settings, get_settings
from db import sqlite_client, supabase_client

logger = logging.getLogger(__name__)

_DATA = Path(__file__).resolve().parent / "data"

# Skip re-embedding / re-upserting scheme vectors if last successful sync within this window.
_SCHEME_VECTOR_SYNC_MIN_INTERVAL_SEC = 7 * 24 * 3600
_SYNC_META_SCHEME_KEY = "schemes_vector_synced_at"


def _build_scheme_vector_batch_from_json(path: Path) -> List[Dict[str, Any]]:
    """CPU-heavy: Chroma embeddings. Intended for asyncio.to_thread."""
    if not path.exists():
        return []
    schemes = json.loads(path.read_text(encoding="utf-8"))
    ef = DefaultEmbeddingFunction()
    batch: List[Dict[str, Any]] = []
    for i, s in enumerate(schemes):
        sid = s.get("id") or f"scheme_{i}"
        text = "\n".join(
            [
                s.get("name", ""),
                s.get("eligibility", ""),
                s.get("benefits", ""),
                s.get("how_to_apply", ""),
                " ".join(s.get("keywords", [])),
            ]
        )
        emb = ef([text])[0]
        batch.append(
            {
                "scheme_id": str(sid),
                "content": text,
                "scheme_json": s,
                "embedding": emb,
            }
        )
    return batch


class SupabaseSync:
    """Idempotent-ish sync job: farmer twin rows → conversation_metadata → query_history → scheme_vectors."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def drain_pending_deletes(self) -> int:
        if not self.settings.supabase_db_configured:
            return 0
        rows = await sqlite_client.fetch_pending_deletes("conversation", self.settings)
        n = 0
        for row in rows:
            row_id = int(row["id"])
            cid = str(row["entity_id"])
            queued_farmer = str(row.get("farmer_id") or "")
            try:
                meta = await supabase_client.get_conversation_metadata_remote(
                    cid, self.settings
                )
                if meta is not None and str(meta.get("farmer_id") or "") != queued_farmer:
                    logger.warning(
                        "dropping pending delete id=%s: farmer_id mismatch for conversation",
                        row_id,
                    )
                    await sqlite_client.remove_pending_delete(row_id, self.settings)
                    continue
                await supabase_client.delete_conversation_remote(cid, self.settings)
                await sqlite_client.remove_pending_delete(row_id, self.settings)
                n += 1
            except Exception as e:
                logger.warning("drain pending delete id=%s: %s", row.get("id"), e)
        return n

    async def sync_farmer_twins(self) -> int:
        if not self.settings.supabase_db_configured:
            return 0
        rows = await sqlite_client.fetch_unsynced_farmer_twins(self.settings)
        n = 0
        for farmer_id, twin in rows:
            try:
                await supabase_client.upsert_farmer_twin_remote(twin, self.settings)
                await sqlite_client.mark_farmer_twin_synced(farmer_id, self.settings)
                n += 1
            except Exception as e:
                logger.warning("sync farmer %s: %s", farmer_id, e)
        return n

    async def sync_conversation_metadata(self) -> int:
        if not self.settings.supabase_db_configured:
            return 0
        rows = await sqlite_client.fetch_unsynced_conversation_metadata(self.settings)
        n = 0
        for row in rows:
            cid = str(row["conversation_id"])
            try:
                await supabase_client.upsert_conversation_metadata_remote(
                    cid,
                    str(row["farmer_id"]),
                    row.get("title"),
                    created_at_unix=int(row["created_at"]),
                    updated_at_unix=int(row["updated_at"]),
                    settings=self.settings,
                )
                await sqlite_client.mark_conversation_metadata_synced(cid, self.settings)
                n += 1
            except Exception as e:
                logger.warning("sync conversation_metadata id=%s: %s", cid, e)
        return n

    async def sync_query_history(self) -> int:
        if not self.settings.supabase_db_configured:
            return 0
        rows = await sqlite_client.fetch_unsynced_query_rows(self.settings)
        n = 0
        for row in rows:
            try:
                await supabase_client.insert_query_history_remote(
                    row["query_text"] or "",
                    row["intent"] or "",
                    row["response"] or "",
                    row["data_source"] or "",
                    sqlite_timestamp_unix=int(row["timestamp"]),
                    conversation_id=row.get("conversation_id"),
                    settings=self.settings,
                )
                await sqlite_client.mark_query_history_synced(int(row["id"]), self.settings)
                n += 1
            except Exception as e:
                logger.warning("sync query id=%s: %s", row.get("id"), e)
        return n

    async def sync_scheme_vectors(self, *, force: bool = False) -> int:
        if not self.settings.supabase_db_configured:
            return 0
        path = _DATA / "scheme_index.json"
        if not path.exists():
            return 0

        now = int(time.time())
        if not force:
            raw = await sqlite_client.get_sync_meta(_SYNC_META_SCHEME_KEY, self.settings)
            if raw:
                try:
                    last = int(raw)
                    if now - last < _SCHEME_VECTOR_SYNC_MIN_INTERVAL_SEC:
                        logger.info(
                            "sync_scheme_vectors: skipped (last sync %s s ago, min interval %s s)",
                            now - last,
                            _SCHEME_VECTOR_SYNC_MIN_INTERVAL_SEC,
                        )
                        return 0
                except ValueError:
                    pass

        batch = await asyncio.to_thread(_build_scheme_vector_batch_from_json, path)
        if not batch:
            return 0
        await supabase_client.upsert_scheme_vector_rows(batch, self.settings)
        await sqlite_client.set_sync_meta(_SYNC_META_SCHEME_KEY, str(now), self.settings)
        return len(batch)

    async def run_without_vectors(self) -> Dict[str, Any]:
        """Sync farmer twin, conversations, query history — not scheme_vectors (heavy)."""
        if not self.settings.supabase_db_configured:
            return {"ok": False, "skipped": True, "reason": "Supabase DB not configured"}
        pending_deleted = await self.drain_pending_deletes()
        farmers = await self.sync_farmer_twins()
        conversations = await self.sync_conversation_metadata()
        queries = await self.sync_query_history()
        return {
            "ok": True,
            "conversation_deletes_drained": pending_deleted,
            "farmer_twins_synced": farmers,
            "conversation_metadata_synced": conversations,
            "query_rows_synced": queries,
            "scheme_chunks_synced": 0,
        }

    async def run(self) -> Dict[str, Any]:
        if not self.settings.supabase_db_configured:
            return {"ok": False, "skipped": True, "reason": "Supabase DB not configured"}
        out = await self.run_without_vectors()
        vectors = await self.sync_scheme_vectors()
        out["scheme_chunks_synced"] = vectors
        return out
