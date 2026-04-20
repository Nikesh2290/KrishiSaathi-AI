"""Query endpoints (POST /query; POST /query/image added in Task 11)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, File, Form, UploadFile

from agent.graph import run_graph
from config.settings import get_settings
from db.sqlite_client import log_query
from models.errors import ErrorCode, KrishiHTTPException
from models.request import AgentRequest
from models.response import AgentResponse
from modules.vision import image_store
from response.generator import build

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["query"])

_ALLOWED_MIME = {"image/jpeg", "image/png"}


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


@router.post("/query/image", status_code=201)
async def post_query_image(
    image: UploadFile = File(...),
    farmer_id: str = Form(...),
    purpose: str = Form(...),
) -> dict:
    settings = get_settings()
    max_bytes = settings.image_max_mb * 1024 * 1024
    data = await image.read()
    if len(data) > max_bytes:
        raise KrishiHTTPException(
            status_code=413,
            code=ErrorCode.IMAGE_TOO_LARGE,
            message=f"Image exceeds {settings.image_max_mb} MB limit.",
        )
    mime = (image.content_type or "").lower()
    if mime not in _ALLOWED_MIME:
        raise KrishiHTTPException(
            status_code=415,
            code=ErrorCode.IMAGE_UNSUPPORTED_TYPE,
            message=f"Unsupported mime '{mime}'. Use JPEG or PNG.",
        )
    ref, expires_at = image_store.put(
        data, mime, farmer_id, purpose, settings.image_upload_ttl_seconds
    )
    return {
        "image_ref": ref,
        "expires_at": datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat(),
        "mime": mime,
        "bytes": len(data),
    }
