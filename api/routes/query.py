"""Query endpoints (POST /query; POST /query/image added in Task 11)."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from agent.graph import run_graph
from config.settings import get_settings
from db.sqlite_client import log_query
from models.request import AgentRequest
from models.response import AgentResponse
from response.generator import build

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["query"])


@router.post("/query", response_model=AgentResponse)
async def post_query(body: AgentRequest) -> AgentResponse:
    settings = get_settings()
    state = await run_graph(body)
    resp = build(
        draft_text=state.get("draft_text") or "",
        tool_results=state.get("tool_results") or {},
        tool_trace=state.get("tool_trace") or [],
        data_source=state.get("data_source") or "live",
        language=body.query.language,
        safety_flags=state.get("safety_flags") or [],
        model_used=state.get("model_used") or settings.ai_studio_model,
        confidence_score=float(state.get("confidence_score") or 0.5),
        fallback_hint=state.get("fallback_hint"),
    )
    try:
        await log_query(
            body.farmer_id,
            body.query.text,
            resp.structured.kind,
            resp.text[:2000],
            resp.data_source,
            settings,
        )
    except Exception as e:
        logger.warning("log_query failed: %s", e)
    return resp
