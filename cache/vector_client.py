"""Upstash Vector async index factory."""

from __future__ import annotations

from typing import Any, Optional

from config.settings import Settings, get_settings


def get_async_vector_index(settings: Optional[Settings] = None) -> Any | None:
    s = settings or get_settings()
    if not s.upstash_vector_configured:
        return None
    from upstash_vector import AsyncIndex

    return AsyncIndex(
        url=s.upstash_vector_rest_url.strip(),
        token=s.upstash_vector_rest_token.strip(),
    )
