"""ChromaDB persistent store over scheme chunks."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

_client: Optional[chromadb.PersistentClient] = None
_collection = None


def _scheme_path() -> Path:
    return Path(__file__).resolve().parents[2] / "offline" / "data" / "scheme_index.json"


def build_index() -> None:
    """Idempotent: (re)build collection from scheme_index.json."""
    global _client, _collection
    settings = get_settings()
    os.makedirs(settings.chroma_path, exist_ok=True)
    _client = chromadb.PersistentClient(path=settings.chroma_path)
    try:
        _client.delete_collection("schemes")
    except Exception:
        pass
    col = _client.get_or_create_collection("schemes", metadata={"hnsw:space": "cosine"})
    path = _scheme_path()
    if not path.exists():
        logger.warning("scheme_index.json missing; empty collection")
        _collection = col
        return
    schemes = json.loads(path.read_text(encoding="utf-8"))
    docs: List[str] = []
    ids: List[str] = []
    metas: List[Dict[str, Any]] = []
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
        docs.append(text)
        ids.append(str(sid))
        metas.append({"name": s.get("name", ""), "json": json.dumps(s, ensure_ascii=False)})
    if docs:
        col.add(ids=ids, documents=docs, metadatas=metas)
    _collection = col
    logger.info("Chroma index built: %s chunks", len(docs))


def get_collection():
    global _client, _collection
    if _collection is not None:
        return _collection
    settings = get_settings()
    os.makedirs(settings.chroma_path, exist_ok=True)
    _client = chromadb.PersistentClient(path=settings.chroma_path)
    try:
        _collection = _client.get_collection("schemes")
    except Exception:
        build_index()
    return _collection


def search(query: str, k: int = 5, *, use_supabase: bool = False) -> List[Dict[str, Any]]:
    """When use_supabase and env is configured, search Supabase pgvector; else ChromaDB."""
    if use_supabase:
        try:
            from db.supabase_client import search_schemes_vector_remote_sync

            remote = search_schemes_vector_remote_sync(query, k=k)
            if remote:
                return remote
        except Exception as e:
            logger.warning("Supabase scheme search failed: %s", e)
    col = get_collection()
    if col.count() == 0:
        return []
    res = col.query(query_texts=[query], n_results=min(k, max(1, col.count())))
    out: List[Dict[str, Any]] = []
    metas = (res.get("metadatas") or [[]])[0]
    for m in metas:
        if not m:
            continue
        js = m.get("json")
        if js:
            out.append(json.loads(js))
    return out


async def search_upstash_async(
    query: str,
    k: int = 5,
    settings: Optional[Settings] = None,
) -> List[Dict[str, Any]]:
    """Gemini embeddings + Upstash Vector cosine search (online)."""
    settings = settings or get_settings()
    if not settings.upstash_vector_configured or not settings.google_api_key.strip():
        return []
    try:
        import asyncio

        from cache import gemini_embed
        from cache.vector_client import get_async_vector_index

        ix = get_async_vector_index(settings)
        if ix is None:
            return []

        vec = await asyncio.to_thread(gemini_embed.embed_query_sync, settings, query or "")
        results = await ix.query(vector=list(vec), top_k=k, include_metadata=True)
        out: List[Dict[str, Any]] = []
        for h in results:
            meta = dict(h.metadata) if h.metadata else {}
            js = meta.get("json")
            if not js:
                continue
            try:
                out.append(json.loads(str(js)))
            except json.JSONDecodeError:
                continue
        return out
    except Exception as e:
        logger.warning("Upstash Vector scheme search failed: %s", e)
        return []
