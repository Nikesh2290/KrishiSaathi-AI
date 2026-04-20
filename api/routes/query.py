"""Query and SSE streaming endpoints."""

from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from agent.orchestrator import run, stream_run
from config.settings import get_settings
from db.sqlite_client import check_rate_limit
from models.request import AgentRequest, ContextPayload, QueryPayload

router = APIRouter(prefix="/api/v1", tags=["query"])


@router.post("/query")
async def post_query(body: AgentRequest):
    settings = get_settings()
    ok, err = await check_rate_limit(body.farmer_id, settings)
    if not ok:
        raise HTTPException(status_code=429, detail=err)
    return await run(body)


@router.get("/query/stream")
async def get_query_stream(
    farmer_id: str = Query(...),
    query_text: str = Query("", alias="q"),
    language: str = Query("hi"),
    connectivity: str = Query("online"),
    lat: float = Query(30.65),
    lng: float = Query(75.95),
    device_intent: str = Query("general"),
):
    settings = get_settings()
    ok, err = await check_rate_limit(farmer_id, settings)
    if not ok:
        raise HTTPException(status_code=429, detail=err)
    req = AgentRequest(
        farmer_id=farmer_id,
        query=QueryPayload(text=query_text, language=language),
        context=ContextPayload(
            location={"lat": lat, "lng": lng},
            connectivity=connectivity,
            device_intent=device_intent,
        ),
    )

    async def gen() -> AsyncIterator[bytes]:
        async for ev in stream_run(req):
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode("utf-8")

    return StreamingResponse(gen(), media_type="text/event-stream")
