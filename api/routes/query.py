"""Query endpoints (POST /query/stream; POST /query/image)."""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import StreamingResponse

from agent.graph import run_graph_stream
from agent.language import LanguageStyle, detect_language_style
from config.settings import get_settings
from db.persistence import persist_log_query, resolve_farmer_twin
from models.errors import ErrorCode, KrishiHTTPException
from models.request import AgentRequest
from modules.vision import image_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["query"])

_ALLOWED_MIME = {"image/jpeg", "image/png"}

# If the user mixes social words with a farming question, do not use the smalltalk fast path.
_FARMING_SUBSTANCE_RE = re.compile(
    r"(?i)\b("
    r"gehu|gahu|gehun|wheat|paddy|chawal|rice|mustard|sarson|tomato|aloo|potato|maize|makka|"
    r"cotton|kapas|sugarcane|ganne|soybean|moong|urad|arhar|tur|chana|lentil|"
    r"bimari|disease|pest|keet|rog|fungus|spray|dawai|khad|fertilizer|urea|dap|npk|"
    r"mandi|market|price|bhav|mausam|weather|barish|rain|sichai|irrigation|"
    r"fasal|crop|field|khet|beej|seed|zaid|rabi|kharif|"
    r"गेहूं|गेहु|फसल|रोग|कीट|मंडी|मौसम|बारिश|खाद|बीज"
    r")\b"
)


_SMALLTALK_GREETING_RE = re.compile(
    r"^\s*(?:(?:hi+|hello+|hey+|hlo+|yo+|namaste|namaskar|नमस्ते|नमस्कार)\b"
    r"(?:\s+(?:ji|krishi|saathi|sathi|bhai|didi|bhaiya)){0,3}|"
    r"good\s*(?:morning|evening|afternoon)|"
    r"(?:kaise\s*ho|kya\s*haal|kaisa\s*ho)\??)\s*[!.,]*\s*$",
    re.IGNORECASE,
)
_SMALLTALK_THANKS_RE = re.compile(
    r"^\s*(?:(?:bahut\s+)+dhanyawad\b|"
    r"(?:thanks?|thank\s*you|thx+|ty+|thnx|dhanyawad|shukriya|धन्यवाद|शुक्रिया|बहुत\s*(?:ही\s*)?धन्यवाद)"
    r"(?:\s+(?:so\s*much|very\s*much|a\s*lot|lot|ji|bahut(?:\s+hi)?|bhaiya|dost)){0,3}"
    r")\s*[!.,]*\s*$",
    re.IGNORECASE,
)
_SMALLTALK_WELCOME_RE = re.compile(
    r"^\s*(?:welcome|you're\s*welcome|ur\s*welcome|आपका\s*स्वागत\s*है|स्वागत\s*है)\s*[!.]*\s*$",
    re.IGNORECASE,
)
_SMALLTALK_BYE_RE = re.compile(
    r"^\s*(?:bye+|good\s*bye|goodbye|see\s*ya|see\s*you(?:\s+later)?|ttyl|gn|good\s*night|"
    r"shubh\s*ratri|शुभ\s*रात्रि|अलविदा|फिर\s*मिलेंगे)"
    r"(?:\s+(?:ji|bhai|dost))?\s*[!.,]*\s*$",
    re.IGNORECASE,
)
_SMALLTALK_ACK_RE = re.compile(
    r"^\s*(?:ok(?:ay)?|okay|k|kk|haan|han|ha|haanji|hmm+|theek\s*hai|ठीक\s*है|अच्छा|sahi|right)"
    r"(?:\s+(?:theek|samajh\s*gaya|got\s*it|sure))?\s*[!.,]*\s*$",
    re.IGNORECASE,
)


def _farmer_display_name(twin_name: str | None) -> str | None:
    n = (twin_name or "").strip()
    return n or None


def _smalltalk_reply(text: str, *, farmer_name: str | None = None) -> tuple[str, str] | None:
    """
    Returns (kind, reply_text) for common one-liner messages
    so we don't spin up the full agent for "hi/thanks/bye/ok".

    ``farmer_name`` from farmer twin is woven into replies when present.
    """
    t = (text or "").strip()
    if not t:
        return None

    if _FARMING_SUBSTANCE_RE.search(t):
        return None

    style: LanguageStyle = detect_language_style(t)
    name = _farmer_display_name(farmer_name)

    def reply_lang(en: str, hi_dev: str, hi_latn: str = "", mixed: str = "") -> str:
        if style == "en":
            return en
        if style == "hi":
            return hi_dev
        if style == "hi-Latn":
            return hi_latn or hi_dev
        return mixed or hi_latn or en

    def greet_hi(en_body: str, hi_body: str, latn_body: str, mix_body: str) -> str:
        if name:
            return reply_lang(
                f"Hi, {name}! {en_body}",
                f"नमस्ते, {name} जी! {hi_body}",
                f"Namaste, {name} ji! {latn_body}",
                f"Hi {name} ji, {mix_body}",
            )
        return reply_lang(
            f"Hi! {en_body}",
            f"नमस्ते! {hi_body}",
            f"Namaste! {latn_body}",
            f"Hi! {mix_body}",
        )

    if _SMALLTALK_GREETING_RE.match(t):
        return (
            "smalltalk_greeting",
            greet_hi(
                "Tell me your crop + district/state and what you want help with—disease, fertilizer, irrigation, mandi price, or schemes.",
                "अपनी फसल का नाम + जिला/राज्य और समस्या बताइए—रोग/कीट, खाद, सिंचाई, मंडी भाव या योजना। मैं तुरंत सही सलाह दूँगा।",
                "Apni fasal + jila/rajya aur samasya batayein—rog, khad, mandi ya yojna. Main turant sahi salah dunga.",
                "Apni crop + district aur problem bataiye — main turant help karunga.",
            ),
        )

    if _SMALLTALK_THANKS_RE.match(t) or _SMALLTALK_WELCOME_RE.match(t):
        thanks_hi = (
            f"{name} जी, आपका स्वागत है। चाहें तो फसल + जिला/राज्य + समस्या एक लाइन में लिख दें — मैं बिल्कुल सटीक सलाह दूँगा।"
            if name
            else "आपका स्वागत है। चाहें तो फसल + जिला/राज्य + समस्या एक लाइन में लिख दें — मैं बिल्कुल सटीक सलाह दूँगा।"
        )
        thanks_en = (
            f"You're welcome, {name}. Share crop + location + one-line issue whenever you like—I’ll give a precise answer."
            if name
            else "You're welcome. Share crop + location + one-line issue whenever you like—I’ll give a precise answer."
        )
        thanks_latn = (
            f"{name} ji, aapka dhanyawad! Jab chahein crop + jila + samasya likh den — main bilkul sahi salah dunga."
            if name
            else "Dhanyawad! Jab chahein crop + jila + samasya likh den — main bilkul sahi salah dunga."
        )
        thanks_mix = (
            f"{name} ji, you're welcome! Crop + location + issue bata dena, main precise answer dunga."
            if name
            else "You're welcome! Crop + location + issue bata dena, main precise answer dunga."
        )
        return (
            "smalltalk_thanks",
            reply_lang(thanks_en, thanks_hi, thanks_latn, thanks_mix),
        )

    if _SMALLTALK_BYE_RE.match(t):
        bye_hi = (
            f"फिर मिलेंगे, {name} जी। फसल + स्थान + समस्या लिख दें तो मैं जल्दी मदद कर दूँगा।"
            if name
            else "ठीक है—फिर मिलेंगे। फसल + स्थान + समस्या लिख दें तो मैं जल्दी मदद कर दूँगा।"
        )
        bye_en = (
            f"Goodbye, {name}. Whenever you're back, share crop + location + issue and I’ll help quickly."
            if name
            else "Goodbye. Whenever you're back, share crop + location + issue and I’ll help quickly."
        )
        bye_latn = (
            f"Phir milenge, {name} ji. Jab wapas aayein to crop + jagaah + samasya likh dena."
            if name
            else "Phir milenge — jab aayein to crop + location + samasya likh dena."
        )
        bye_mix = (
            f"Bye {name} ji! Wapas aate hi crop + location bata dena — main help kar dunga."
            if name
            else "Bye! Wapas aate hi boliye — main help kar dunga."
        )
        return ("smalltalk_bye", reply_lang(bye_en, bye_hi, bye_latn, bye_mix))

    if _SMALLTALK_ACK_RE.match(t):
        ack_hi = (
            f"ठीक है, {name} जी। अब क्या चाहिए—रोग/कीट, खाद‑प्लान, सिंचाई, मंडी भाव या सरकारी योजना?"
            if name
            else "ठीक है। अब क्या चाहिए—रोग/कीट, खाद‑प्लान, सिंचाई, मंडी भाव या सरकारी योजना?"
        )
        ack_en = (
            f"Got it, {name}. What next—disease/pest, fertilizer plan, irrigation, mandi price, or a scheme?"
            if name
            else "Got it. What next—disease/pest, fertilizer plan, irrigation, mandi price, or a scheme?"
        )
        ack_latn = (
            f"Theek hai, {name} ji. Ab kya chahiye—rog, khad, mandi bhav ya yojna?"
            if name
            else "Theek hai. Ab kya chahiye—rog, khad, mandi bhav ya yojna?"
        )
        ack_mix = (
            f"Got it {name} ji! Ab bataiye — disease, fertilizer, mandi price, ya koi scheme?"
            if name
            else "Got it! Ab bataiye kya chahiye."
        )
        return ("smalltalk_ack", reply_lang(ack_en, ack_hi, ack_latn, ack_mix))

    return None


@router.post("/query/stream")
async def post_query_stream(body: AgentRequest) -> StreamingResponse:
    async def event_stream():
        try:
            if not body.query.image_ref:
                settings = get_settings()
                twin = None
                try:
                    twin = await resolve_farmer_twin(
                        body.farmer_id, body.context.connectivity, settings
                    )
                except Exception as e:
                    logger.warning("resolve_farmer_twin for smalltalk failed: %s", e)

                farmer_nm = twin.name if twin else None
                st = _smalltalk_reply(body.query.text or "", farmer_name=farmer_nm)
                if st:
                    _kind, reply = st
                    message_id = str(uuid.uuid4())
                    text_id = f"txt_{uuid.uuid4().hex}"
                    try:
                        await persist_log_query(
                            body.query.text,
                            _kind,
                            reply[:2000],
                            "live",
                            body.context.connectivity,
                            farmer_id=body.farmer_id,
                            conversation_id=body.conversation_id,
                            settings=settings,
                        )
                    except Exception as e:
                        logger.warning("log_query failed: %s", e)

                    yield f"data: {json.dumps({'type': 'start', 'messageId': message_id}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type': 'data-tool', 'data': {'tool': 'smalltalk', 'status': 'started'}}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type': 'start-step'}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type': 'text-start', 'id': text_id}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type': 'text-delta', 'id': text_id, 'delta': reply}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type': 'text-end', 'id': text_id}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type': 'finish-step'}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type': 'data-tool', 'data': {'tool': 'smalltalk', 'status': 'done'}}, ensure_ascii=False)}\n\n"
                    meta = {"type": "data-metadata", "data": {"conversation_id": body.conversation_id}}
                    yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type': 'finish'}, ensure_ascii=False)}\n\n"
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
        except KrishiHTTPException as exc:
            err_obj = {
                "type": "error",
                "errorText": str(exc.detail),
                "errorCode": exc.code.value,
                "statusCode": exc.status_code,
            }
            yield f"data: {json.dumps(err_obj, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
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
