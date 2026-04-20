"""Orchestrator: graph + safety + structured response."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, List

from agent.react_loop import run_graph, stream_events
from config.settings import get_settings
from db.sqlite_client import log_query
from models.request import AgentRequest
from models.response import AgentResponse
from response.generator import build
from safety.layer import check

logger = logging.getLogger(__name__)


def _vision_confidence(tool_results: Dict[str, Any]) -> float | None:
    for v in tool_results.values():
        if isinstance(v, dict) and "confidence" in v:
            return float(v["confidence"])
    return None


async def run(req: AgentRequest) -> AgentResponse:
    settings = get_settings()
    state = await run_graph(req)
    draft = state.get("draft_text") or ""
    tool_results = state.get("tool_results") or {}
    tool_trace = state.get("tool_trace") or []
    data_source = state.get("data_source") or "live"
    vc = _vision_confidence(tool_results)
    sr = check(tool_results, draft, vision_confidence=vc)
    resp = build(
        sr.modified_text,
        tool_results,
        tool_trace,
        data_source,
        req.query.language,
        sr.flags,
    )
    try:
        await log_query(
            req.farmer_id,
            req.query.text,
            resp.structured.kind,
            resp.text[:2000],
            resp.data_source,
            settings,
        )
    except Exception as e:
        logger.warning("log_query failed: %s", e)
    return resp


async def stream_run(req: AgentRequest) -> AsyncIterator[Dict[str, Any]]:
    async for ev in stream_events(req):
        yield ev
