"""Public cache warmup + admin hooks."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Header, HTTPException
from pydantic import BaseModel, Field

from cache.warmup import run_warmup, scopes_for_cron
from config.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["cache"])


class WarmupRequest(BaseModel):
    force: bool = Field(True, description="Always run (reserved for conditional warm in future)")
    scopes: Optional[List[str]] = Field(
        default=None,
        description="Subset: weather, mandi, schemes_index, schemes_vector",
    )
    freq: Optional[str] = Field(
        default=None,
        description='If scopes empty, presets: "frequent" | "daily_vectors"',
    )
    include_scheme_vectors: bool = Field(
        default=False,
        description="If true and scopes include schemes_vector, run Gemini embedding upserts.",
    )


@router.post("/cache/warmup")
async def post_cache_warmup(
    body: WarmupRequest = Body(default_factory=WarmupRequest),
    x_warmup_secret: Optional[str] = Header(None, alias="X-Warmup-Secret"),
) -> Dict[str, Any]:
    settings = get_settings()
    expected = settings.warmup_secret.strip()
    if expected and (x_warmup_secret or "").strip() != expected:
        raise HTTPException(status_code=401, detail="warmup unauthorized")

    scopes_eff = body.scopes
    if body.freq and not scopes_eff:
        scopes_eff = scopes_for_cron(body.freq)
    stats = await run_warmup(
        settings,
        scopes=scopes_eff,
        include_scheme_vectors=bool(body.include_scheme_vectors),
    )

    return stats
