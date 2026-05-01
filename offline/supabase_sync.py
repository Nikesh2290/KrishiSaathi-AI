"""Push offline SQLite data and scheme embeddings to Supabase (Postgres + pgvector)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from config.settings import Settings, get_settings
from db import sqlite_client, supabase_client

logger = logging.getLogger(__name__)

_DATA = Path(__file__).resolve().parent / "data"


class SupabaseSync:
    """Idempotent-ish sync job: farmer twin rows → query_history → scheme_vectors."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

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

    async def sync_query_history(self) -> int:
        if not self.settings.supabase_db_configured:
            return 0
        rows = await sqlite_client.fetch_unsynced_query_rows(self.settings)
        n = 0
        for row in rows:
            try:
                await supabase_client.insert_query_history_remote(
                    str(row["farmer_id"]),
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

    async def sync_scheme_vectors(self) -> int:
        if not self.settings.supabase_db_configured:
            return 0
        path = _DATA / "scheme_index.json"
        if not path.exists():
            return 0
        schemes = json.loads(path.read_text(encoding="utf-8"))
        ef = DefaultEmbeddingFunction()
        batch: list[Dict[str, Any]] = []
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
        if not batch:
            return 0
        await supabase_client.upsert_scheme_vector_rows(batch, self.settings)
        return len(batch)

    async def run(self) -> Dict[str, Any]:
        if not self.settings.supabase_db_configured:
            return {"ok": False, "skipped": True, "reason": "Supabase DB not configured"}
        farmers = await self.sync_farmer_twins()
        queries = await self.sync_query_history()
        vectors = await self.sync_scheme_vectors()
        return {
            "ok": True,
            "farmer_twins_synced": farmers,
            "query_rows_synced": queries,
            "scheme_chunks_synced": vectors,
        }
