"""Query endpoints (POST /query; POST /query/image added in Task 11)."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import StreamingResponse

from agent.graph import run_graph, run_graph_stream
from config.settings import get_settings
from db.persistence import persist_log_query
from models.errors import ErrorCode, KrishiHTTPException
from models.request import AgentRequest
from models.response import AgentResponse, StructuredResult
from modules.vision import image_store
from response.generator import build

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["query"])

_ALLOWED_MIME = {"image/jpeg", "image/png"}

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

_SMALLTALK_GREETING_RE = re.compile(
    r"^\s*(?:hi+|hello+|hey+|hlo+|yo+|namaste|namaskar|नमस्ते|नमस्कार)\s*[!.]*\s*$",
    re.IGNORECASE,
)
_SMALLTALK_THANKS_RE = re.compile(
    r"^\s*(?:thanks|thank\s*you|thx+|ty+|धन्यवाद|शुक्रिया|बहुत\s*धन्यवाद|thnx)\s*[!.]*\s*$",
    re.IGNORECASE,
)
_SMALLTALK_WELCOME_RE = re.compile(
    r"^\s*(?:welcome|you're\s*welcome|ur\s*welcome|आपका\s*स्वागत\s*है|स्वागत\s*है)\s*[!.]*\s*$",
    re.IGNORECASE,
)
_SMALLTALK_BYE_RE = re.compile(
    r"^\s*(?:bye+|good\s*bye|goodbye|see\s*ya|see\s*you|ttyl|gn|good\s*night|shubh\s*ratri|शुभ\s*रात्रि|अलविदा|फिर\s*मिलेंगे)\s*[!.]*\s*$",
    re.IGNORECASE,
)
_SMALLTALK_ACK_RE = re.compile(
    r"^\s*(?:ok(?:ay)?|okay|k|kk|haan|han|ha|haanji|hmm+|theek\s*hai|ठीक\s*है|अच्छा|sahi|right)\s*[!.]*\s*$",
    re.IGNORECASE,
)


def _smalltalk_reply(text: str) -> tuple[str, str] | None:
    """
    Returns (kind, reply_text) for common one-liner messages
    so we don't spin up the full agent for "hi/thanks/bye/ok".
    """
    t = (text or "").strip()
    if not t:
        return None

    is_hi = bool(_DEVANAGARI_RE.search(t))

    def hi(en: str, hi_msg: str) -> str:
        return hi_msg if is_hi else en

    if _SMALLTALK_GREETING_RE.match(t):
        return (
            "smalltalk_greeting",
            hi(
                "Hi! Tell me your crop + district/state and what you want help with (disease, fertilizer, irrigation, price, scheme).",
                "नमस्ते! अपनी फसल का नाम + जिला/राज्य और समस्या लिखिए (रोग/कीट, खाद, सिंचाई, मंडी भाव, योजना) — मैं तुरंत सही सलाह दूँगा।",
            ),
        )

    if _SMALLTALK_THANKS_RE.match(t) or _SMALLTALK_WELCOME_RE.match(t):
        return (
            "smalltalk_thanks",
            hi(
                "You're welcome. If you want, share crop + location + one-line issue and I’ll give a precise step-by-step answer.",
                "आपका स्वागत है। चाहें तो फसल + जिला/राज्य + समस्या एक लाइन में लिख दें — मैं बिल्कुल सटीक, स्टेप‑बाय‑स्टेप सलाह दे दूँगा।",
            ),
        )

    if _SMALLTALK_BYE_RE.match(t):
        return (
            "smalltalk_bye",
            hi(
                "Goodbye. Whenever you return, share crop + location + issue (or upload a leaf photo) and I’ll help quickly.",
                "ठीक है—फिर मिलेंगे। कभी भी आएँ, फसल + स्थान + समस्या लिख दें (या पत्ते की फोटो) — मैं जल्दी मदद कर दूँगा।",
            ),
        )

    if _SMALLTALK_ACK_RE.match(t):
        return (
            "smalltalk_ack",
            hi(
                "Got it. What do you want next—disease/pest, fertilizer plan, irrigation timing, mandi price, or a government scheme?",
                "ठीक है। अब आपको क्या चाहिए—रोग/कीट, खाद‑उर्वरक प्लान, सिंचाई समय, मंडी भाव, या कोई सरकारी योजना?",
            ),
        )

    return None


@router.post("/query", response_model=AgentResponse)
async def post_query(body: AgentRequest) -> AgentResponse:
    settings = get_settings()
    if not body.query.image_ref:
        st = _smalltalk_reply(body.query.text or "")
        if st:
            kind, reply = st
            return AgentResponse(
                text=reply,
                structured=StructuredResult(kind=kind, data={"intent": kind}),
                data_source="live",
                confidence_level="high",
                confidence_score=0.99,
                model_used="rules",
                tool_trace=["smalltalk"],
                language=body.query.language,
            )
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
        await persist_log_query(
            body.farmer_id,
            body.query.text,
            resp.structured.kind,
            resp.text[:2000],
            resp.data_source,
            body.context.connectivity,
            settings,
        )
    except Exception as e:
        logger.warning("log_query failed: %s", e)
    return resp


@router.post("/query/stream")
async def post_query_stream(body: AgentRequest) -> StreamingResponse:
    async def event_stream():
        try:
            if not body.query.image_ref:
                st = _smalltalk_reply(body.query.text or "")
                if st:
                    _kind, reply = st
                    payload: dict[str, object] = {"type": "text-delta", "delta": reply}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
            async for event_type, data in run_graph_stream(body):
                if event_type == "__done__":
                    yield "data: [DONE]\n\n"
                else:
                    payload: dict[str, object] = {"type": event_type}
                    if data is not None:
                        payload.update(data)
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.exception("query/stream failed: %s", e)
            err_payload = json.dumps({"type": "error", "errorText": str(e)})
            yield f"data: {err_payload}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "x-vercel-ai-ui-message-stream": "v1",
        },
    )


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
