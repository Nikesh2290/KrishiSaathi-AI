"""Google Gemini text embeddings for Upstash Vector (768 dims for text-embedding-004)."""

from __future__ import annotations

import logging
from typing import List

from config.settings import Settings

logger = logging.getLogger(__name__)


def embed_query_sync(settings: Settings, text: str) -> List[float]:
    if not settings.google_api_key.strip():
        raise RuntimeError("GOOGLE_AI_STUDIO_KEY required for Gemini embeddings")

    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except ImportError as e:
        raise RuntimeError("langchain-google-genai is required for embeddings") from e

    model_name = settings.google_embedding_model.strip()
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"
    ef = GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=settings.google_api_key,
    )
    vec = ef.embed_query(text or "")
    return [float(x) for x in vec]


def embed_documents_sync(settings: Settings, texts: List[str]) -> List[List[float]]:
    if not settings.google_api_key.strip():
        raise RuntimeError("GOOGLE_AI_STUDIO_KEY required for Gemini embeddings")

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    model_name = settings.google_embedding_model.strip()
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"
    ef = GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=settings.google_api_key,
    )
    out = ef.embed_documents([t or "." for t in texts])
    return [[float(x) for x in row] for row in out]
