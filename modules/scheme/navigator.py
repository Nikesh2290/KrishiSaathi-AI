"""RAG answer for schemes using retrieval + Gemma narration."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from agent.gemma_client import generate
from config.settings import Settings, get_settings
from models.farmer import FarmerTwin
from modules.scheme import offline_search, vector_store

logger = logging.getLogger(__name__)


async def find_schemes(
    farmer: FarmerTwin | None,
    query: str,
    prefer_local: bool,
    offline_mode: bool,
    settings: Settings | None = None,
) -> Dict[str, Any]:
    settings = settings or get_settings()
    retrieved: list = []
    if offline_mode:
        retrieved = offline_search.keyword_search(query, limit=5)
        source = "offline_keyword"
    else:
        try:
            retrieved = vector_store.search(query, k=5)
            source = "chroma"
        except Exception as e:
            logger.warning("Chroma search failed: %s", e)
            retrieved = offline_search.keyword_search(query, limit=5)
            source = "offline_keyword_fallback"

    context = json.dumps(retrieved, ensure_ascii=False)[:8000]
    loc = farmer.location if farmer else None
    loc_s = f"{loc.state}/{loc.district}" if loc else "unknown"
    messages = [
        {
            "role": "system",
            "content": "You help Indian farmers understand government schemes. Answer in the user's language (Hindi if query is Hindi). Cite scheme names. If unsure, say consult nearest CSC.",
        },
        {
            "role": "user",
            "content": f"Farmer region: {loc_s}. Question: {query}\n\nSchemes data:\n{context}",
        },
    ]
    text = await generate(messages, prefer_local=prefer_local, settings=settings)
    return {"answer": text, "schemes": retrieved, "source": source}
