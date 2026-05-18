"""LiveKit Agents entrypoint: STT + HTTP Krishi /query/stream + TTS."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import uuid
from pathlib import Path
from collections.abc import AsyncIterable
from typing import Any, Optional

from dotenv import load_dotenv

import httpx
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.voice import ModelSettings, room_io
from livekit.plugins import deepgram, silero

from agent.language import detect_language_style, filler_lang_key
from config.settings import get_settings
from db.persistence import resolve_farmer_twin
from models.farmer import FarmerTwin
from voice_agent.query_stream_client import iter_query_stream_events
from voice_agent.stub_llm import KrishiStubLLM
from voice_agent.timing import VoiceTimeline, new_turn_id, voice_timing_enabled

# LiveKit CLI reads os.environ before connecting; load repo .env regardless of cwd.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

logger = logging.getLogger(__name__)

# LiveKit may send control text streams on this topic; drain them so the RTC layer
# does not log "no callback attached" and streams complete cleanly.
_LK_AGENT_REQUEST_TOPIC = "lk.agent.request"

_WELCOME_SENT_KEYS: set[str] = set()
_VOICE_MIN_CHUNK_CHARS = max(8, int(os.getenv("VOICE_MIN_CHUNK_CHARS", "10")))
_VOICE_CHUNK_MAX_BUF = max(120, int(os.getenv("VOICE_CHUNK_MAX_BUF", "200")))
_MAX_FILLERS_PER_TURN = 3

_TOOL_COUNT_TOOLS = frozenset(
    {
        "climate",
        "market",
        "scheme",
        "crop_planner",
        "financial",
        "vision",
        "general_qa",
    }
)


def _register_lk_agent_request_sink(room: rtc.Room) -> None:
    def _on_text(reader: rtc.TextStreamReader, _participant_identity: str) -> None:
        async def _drain() -> None:
            try:
                await reader.read_all()
            except Exception:
                logger.debug("lk.agent.request stream ended", exc_info=True)

        asyncio.create_task(_drain())

    try:
        room.register_text_stream_handler(_LK_AGENT_REQUEST_TOPIC, _on_text)
    except ValueError:
        # Handler already registered (e.g. reconnect) — keep the first one.
        pass


def _last_user_message_text(chat_ctx: llm.ChatContext) -> str:
    from livekit.agents.llm import ChatMessage

    # ChatRole is Literal["developer","system","user","assistant"], not an enum.
    for it in reversed(chat_ctx.items):
        if isinstance(it, ChatMessage) and it.role == "user":
            return (it.text_content or "").strip()
    return ""


def _parse_farmer_meta(metadata: str) -> dict[str, Any]:
    if not (metadata or "").strip():
        return {}
    try:
        return json.loads(metadata)
    except json.JSONDecodeError:
        logger.warning("participant metadata is not valid JSON")
        return {}


def _deepgram_stt_language(locale: str) -> str:
    """Single language for Deepgram live streaming (detect_language is not supported)."""
    override = os.getenv("DEEPGRAM_STT_LANGUAGE", "").strip()
    if override:
        return override
    raw = (locale or "en").strip().replace("_", "-").lower()
    if not raw or raw == "und":
        return "hi"
    base = raw.split("-", 1)[0]
    if base in ("hi", "hin") and ("latn" in raw or "latin" in raw):
        return "hi-Latn"
    if base in ("hi", "hin"):
        return "hi"
    if base in ("en", "eng"):
        return "en-IN" if len(raw.split("-")) == 1 or "in" in raw.split("-") else "en-US"
    if base in ("ta", "tam"):
        return "ta"
    if base in ("taq",):
        return "taq"
    # Other Indian ISO codes: Deepgram nova streaming list varies — force explicit override or Hindi.
    if base in ("te", "kn", "mr", "bn", "gu", "pa", "ml", "or", "as", "ur"):
        logger.warning(
            "voice locale %r has no built-in Deepgram STT mapping; using hi "
            "(set DEEPGRAM_STT_LANGUAGE for a supported code).",
            locale,
        )
        return "hi"
    if base in ("multi", "mul"):
        return "hi"
    return base if len(base) == 2 else "hi"


def _sentence_boundary(buf: str) -> int | None:
    """Exclusive index to flush a spoken phrase (sentence-ish chunk)."""
    candidates: list[int] = []
    for sep in ("।", "\n"):
        i = buf.find(sep)
        if i != -1:
            candidates.append(i + len(sep))
    for sep in ("!", "?"):
        i = buf.find(sep)
        if i != -1:
            candidates.append(i + len(sep))
    i = buf.find(".")
    while i != -1:
        if i > 0 and i + 1 < len(buf) and buf[i - 1].isdigit() and buf[i + 1].isdigit():
            i = buf.find(".", i + 1)
            continue
        candidates.append(i + len("."))
        break
    return min(candidates) if candidates else None


def _is_strong_flush_boundary(fragment: str) -> bool:
    t = fragment.rstrip()
    if not t:
        return False
    if t.endswith(("।", "!", "?", "\n")):
        return True
    if t.endswith("."):
        # Avoid flushing tiny "Dr." chunks as strong
        return len(t) >= _VOICE_MIN_CHUNK_CHARS
    return False


def _strip_chunk_overlap(chunk: str, prev_tail: str, *, min_ov: int = 4, max_ov: int = 24) -> str:
    if not chunk or not prev_tail:
        return chunk
    tail = prev_tail[-120:]
    upper = min(max_ov, len(chunk), len(tail))
    for n in range(upper, min_ov - 1, -1):
        if chunk[:n] == tail[-n:]:
            return chunk[n:].lstrip()
    return chunk


def _format_filler(template: str, name: str | None) -> str:
    n = (name or "").strip()
    if n:
        return template.replace("{name}", n)
    s = template
    for pat in ("{name} ji, ", "{name} ji ", "{name}, ", "{name} ", "{name}"):
        s = s.replace(pat, "")
    return re.sub(r"\s+", " ", s).strip()


def _pick_variant(variants: list[str], name: str | None, spoken: set[str]) -> str | None:
    formatted = [_format_filler(v, name) for v in variants]
    candidates = [f for f in formatted if f and f not in spoken]
    if not candidates:
        return None
    choice = random.choice(candidates)
    spoken.add(choice)
    return choice


# Stage-level phrases (empty = no filler); ``tools`` is silent — per-tool lines handle UX.
_STAGE_PHRASE_LISTS: dict[str, dict[str, list[str]]] = {
    "hi": {
        "routing": [],
        "smalltalk": [],
        "direct_llm": [],
        "clarify": [],
        "tools": [],
        "synthesizing": [],
    },
    "en": {
        "routing": [],
        "smalltalk": [],
        "direct_llm": [],
        "clarify": [],
        "tools": [],
        "synthesizing": [],
    },
}

_TOOL_PHRASE_LISTS: dict[str, dict[str, list[str]]] = {
    "hi": {
        "thinking": [
            "Haan {name} ji, aapka sawaal samajh ke seedha jawaab de raha hoon — bas ek second।",
            "{name} ji, soch raha hoon — thodi si der mein poori baat bata deta hoon।",
        ],
        "climate": [
            "{name} ji, aapke ilake ka mausam aur aane waali barish ki jaankari dekh raha hoon, taaki sahi salaah de sakoon।",
            "Ek second, {name} ji — aapke jile ka aaj ka mausam aur agli kuch dinon ka anumaan dekh raha hoon।",
        ],
        "market": [
            "{name} ji, aapki fasal ke liye najdeeki mandion ke aaj ke live bhaav dekh raha hoon।",
            "Haan ji, {name} ji — mandi ke taza bhaav nikaal raha hoon, bas thodi der mein bata deta hoon।",
        ],
        "scheme": [
            "{name} ji, aapki fasal aur zameen ke hisaab se kaunsi sarkari yojnaaon ka labh milega, woh dhundh raha hoon — PM Kisan, KCC aur baaki।",
            "Ek second, {name} ji — aapke liye central aur state dono ki madad kya milti hai, yeh dekh raha hoon।",
        ],
        "crop_planner": [
            "{name} ji, aapki fasal, zameen aur mausam ko dhyan mein rakhte hue poori aur seedhi salaah bana raha hoon।",
            "Ek second — aapke kshetra ke liye beej chunaav se kaataai tak ka pura plan tayyar kar raha hoon, {name} ji।",
        ],
        "financial": [
            "{name} ji, aapke liye Kisan Credit Card, fasal beema aur subsidy ki puri jaankari ek jagah kar raha hoon।",
            "Ek second, {name} ji — loan ki limit, byaaj dar aur beema claim ka tarika — sab milaakar clearly bata deta hoon।",
        ],
        "vision": [
            "{name} ji, aapki tasveer dhyan se dekh raha hoon — fasal mein rog ya keede ki sahi pehchaan kar raha hoon।",
            "Ek second, {name} ji — photo mein kya samasya hai, identify karke poora hal bata deta hoon।",
        ],
        "general_qa": [
            "{name} ji, seedha jawab dhoondh raha hoon — ek second mein bilkul sahi baat bata deta hoon।",
        ],
        "synthesizing": [
            "Sab jaankari aa gayi, {name} ji — ab aapka poora aur seedha jawaab tayyar kar raha hoon।",
            "{name} ji, data aa gaya — sab kuch milaakar ek baar mein clearly samjhata hoon।",
        ],
    },
    "en": {
        "thinking": [
            "{name}, let me think through your question carefully — I'll have a complete answer for you in just a moment.",
            "Just a second, {name} — working through this to give you the most accurate answer I can.",
        ],
        "climate": [
            "{name}, I'm checking the latest weather forecast and rainfall data for your area so I can give you the right advice.",
            "One moment, {name} — pulling up current conditions and the next few days' outlook for your district.",
        ],
        "market": [
            "{name}, I'm checking today's live mandi prices for your crop at the nearest markets — just a second.",
            "Just a moment, {name} — fetching the latest market rates so you get the most current prices.",
        ],
        "scheme": [
            "{name}, I'm checking which central and state government schemes you're eligible for based on your crop and location — just a moment.",
        ],
        "crop_planner": [
            "{name}, I'm preparing a detailed crop plan based on your soil type, local weather, and the current season — just a second.",
        ],
        "financial": [
            "{name}, I'm looking up Kisan Credit Card limits, crop insurance options, and available subsidies for you — just a moment.",
        ],
        "vision": [
            "{name}, I'm carefully analyzing your crop photo to identify the disease or pest — I want to make sure I give you the right diagnosis.",
        ],
        "general_qa": [
            "{name}, I'm finding a direct answer for you — just a moment.",
        ],
        "synthesizing": [
            "Got all the data, {name} — now putting together a clear, complete answer for you.",
            "{name}, the information is in — let me compile it all into a proper response.",
        ],
    },
}


def _build_voice_welcome(twin: FarmerTwin | None, session_lang: str) -> str:
    """Rich tiered welcome using twin profile (name, location, crops, land)."""
    base = (session_lang or "en").split("-", 1)[0].lower()
    is_hi = base not in ("en", "eng")

    name = (twin.name.strip() if twin and twin.name else "") or None
    district = (twin.location.district.strip() if twin and twin.location.district else "") or None
    state = (twin.location.state.strip() if twin and twin.location.state else "") or None
    crops = [c.strip() for c in (twin.current_crops if twin else []) if c.strip()]
    acres = float(twin.land.total_acres) if twin and twin.land.total_acres > 0 else None

    location_str = f"{district}, {state}" if district and state else district or state or None
    crop_hi = " और ".join(crops[:2])
    crop_en = " and ".join(crops[:2])

    if name and location_str and crops:
        acres_part_hi = f"आपके {acres:.0f} एकड़ खेत में " if acres else ""
        acres_part_en = f"your {acres:.0f}-acre farm with " if acres else "your "
        if is_hi:
            return (
                f"नमस्ते, {name} जी! मैं Krishi Saathi हूँ — आपका अपना खेती सलाहकार। "
                f"{location_str} में {acres_part_hi}{crop_hi} की देखभाल के लिए मैं हमेशा तैयार हूँ। "
                f"मौसम, मंडी भाव, रोग-कीट, खाद, या सरकारी योजना — जो भी पूछना हो, बताइए।"
            )
        return (
            f"Hello, {name}! I'm Krishi Saathi, your personal farming assistant. "
            f"I'm here to help with {acres_part_en}{crop_en} in {location_str} — "
            f"whether it's weather, mandi prices, disease, fertilizers, or government schemes. "
            f"What would you like to know today?"
        )

    if name and location_str:
        if is_hi:
            return (
                f"नमस्ते, {name} जी! मैं Krishi Saathi हूँ — {location_str} के किसानों की मदद करना मैं अपना काम समझता हूँ। "
                f"मौसम, मंडी भाव, रोग, खाद-प्लान, या सरकारी योजना — कुछ भी पूछिए, मैं यहाँ हूँ।"
            )
        return (
            f"Hello, {name}! I'm Krishi Saathi, your farming assistant for the {location_str} region. "
            f"Ask me anything — weather, mandi prices, crop disease, fertilizers, or government schemes."
        )

    if name:
        if is_hi:
            return (
                f"नमस्ते, {name} जी! मैं Krishi Saathi हूँ — आपका खेती सलाहकार। "
                f"फसल की कोई भी समस्या हो — रोग, मौसम, मंडी भाव, या सरकारी योजना — बस बताइए।"
            )
        return (
            f"Hello, {name}! I'm Krishi Saathi, your personal farming assistant. "
            f"Ask me about crop disease, weather, mandi prices, or government schemes — I'm here to help."
        )

    if is_hi:
        return (
            "नमस्ते! मैं Krishi Saathi हूँ — आपका अपना खेती सलाहकार। "
            "फसल का नाम और अपना जिला बताइए — मैं आपकी पूरी मदद करूँगा।"
        )
    return (
        "Hello! I'm Krishi Saathi, your personal farming assistant. "
        "Tell me your crop and district, and I'll help you with weather, prices, disease, or schemes."
    )


class KrishiVoiceAgent(Agent):
    def __init__(
        self,
        *,
        api_base: str,
        farmer_id: str,
        conversation_id: Optional[str],
        language: str,
        http_client: httpx.AsyncClient,
        farmer_name: str | None = None,
        session_id: str | None = None,
        room_name: str | None = None,
    ) -> None:
        self._api_base = api_base.rstrip("/")
        self._farmer_id = farmer_id
        self._conversation_id = conversation_id
        self._language = language or "en"
        self._farmer_name = farmer_name
        self._session_id = session_id
        self._room_name = room_name
        self._http_client = http_client
        super().__init__(
            instructions=(
                "You are Krishi Saathi, an Indian agriculture assistant. "
                "Answers are produced by the Krishi backend from the farmer's speech."
            ),
            llm=KrishiStubLLM(),
        )

    def _timeline(self, turn_id: str) -> VoiceTimeline:
        return VoiceTimeline(
            turn_id=turn_id,
            session_id=self._session_id,
            farmer_id=self._farmer_id,
            conversation_id=self._conversation_id,
            room=self._room_name,
        )

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[str]:
        del tools, model_settings
        turn_id = new_turn_id()
        tl = self._timeline(turn_id)
        tl.mark("llm_node_start", voice_timing=voice_timing_enabled())

        user_text = _last_user_message_text(chat_ctx)
        if not user_text:
            tl.mark("llm_node_abort", reason="empty_user_text")
            yield "Please say your farming question again."
            return

        tl.mark(
            "stt_text_ready",
            user_text_len=len(user_text),
            user_text_preview=user_text[:120],
        )

        detected = detect_language_style(user_text)
        query_lang: str = detected
        tl.mark("language_detected", query_lang=query_lang, detected_style=detected)

        payload: dict[str, Any] = {
            "farmer_id": self._farmer_id,
            "conversation_id": self._conversation_id,
            "query": {"text": user_text, "language": query_lang},
            "context": {
                "connectivity": "online",
                "location": {},
                "device_intent": "voice",
                "device_capabilities": {},
            },
        }

        lang_key = filler_lang_key(detected, self._language)
        if lang_key not in _STAGE_PHRASE_LISTS:
            lang_key = "hi"
        stage_lists = _STAGE_PHRASE_LISTS[lang_key]
        tool_lists = _TOOL_PHRASE_LISTS[lang_key]
        spoken_fillers: set[str] = set()
        filler_count = 0
        tool_only_starts = 0

        def _try_emit_tool_filler(tool: str) -> str | None:
            nonlocal filler_count
            if filler_count >= _MAX_FILLERS_PER_TURN:
                return None
            variants = tool_lists.get(tool) or []
            if not variants:
                return None
            return _pick_variant(variants, self._farmer_name, spoken_fillers)

        sentence_buf = ""
        saw_answer_delta = False
        last_spoken_tail = ""
        tts_chunk_count = 0
        filler_events = 0
        sse_event_count = 0

        try:
            tl.mark("api_query_stream_start", api_base=self._api_base)
            async for obj in iter_query_stream_events(
                self._http_client,
                api_base=self._api_base,
                payload=payload,
                timeline=tl,
                turn_id=turn_id,
            ):
                sse_event_count += 1
                otype = obj.get("type")
                if otype == "data-stage":
                    data = obj.get("data") if isinstance(obj.get("data"), dict) else {}
                    stage = str(data.get("stage") or "")
                    variants = stage_lists.get(stage) or []
                    if saw_answer_delta or filler_count >= _MAX_FILLERS_PER_TURN:
                        continue
                    phrase = _pick_variant(variants, self._farmer_name, spoken_fillers)
                    if phrase:
                        filler_count += 1
                        filler_events += 1
                        tl.mark(
                            "filler_yield_stage",
                            stage=stage,
                            phrase_chars=len(phrase),
                            filler_index=filler_count,
                        )
                        yield phrase + " "
                elif otype == "data-tool":
                    data = obj.get("data") if isinstance(obj.get("data"), dict) else {}
                    if data.get("status") != "started":
                        continue
                    tool = str(data.get("tool") or "")
                    if tool in _TOOL_COUNT_TOOLS:
                        tool_only_starts += 1
                    if saw_answer_delta:
                        continue
                    if tool == "synthesizing":
                        if tool_only_starts >= 2 and filler_count < _MAX_FILLERS_PER_TURN:
                            phrase = _try_emit_tool_filler("synthesizing")
                            if phrase:
                                filler_count += 1
                                filler_events += 1
                                tl.mark(
                                    "filler_yield_tool",
                                    tool=tool,
                                    phrase_chars=len(phrase),
                                    filler_index=filler_count,
                                )
                                yield phrase + " "
                        continue
                    if tool in _TOOL_COUNT_TOOLS and filler_count < 1:
                        phrase = _try_emit_tool_filler(tool)
                        if phrase:
                            filler_count += 1
                            filler_events += 1
                            tl.mark(
                                "filler_yield_tool",
                                tool=tool,
                                phrase_chars=len(phrase),
                                filler_index=filler_count,
                            )
                            yield phrase + " "
                    elif tool == "thinking" and filler_count < 1:
                        phrase = _try_emit_tool_filler("thinking")
                        if phrase:
                            filler_count += 1
                            filler_events += 1
                            tl.mark(
                                "filler_yield_tool",
                                tool="thinking",
                                phrase_chars=len(phrase),
                                filler_index=filler_count,
                            )
                            yield phrase + " "
                elif otype == "text-delta":
                    delta = str(obj.get("delta") or "")
                    if not delta:
                        continue
                    if not saw_answer_delta:
                        tl.mark("first_answer_token", delta_chars=len(delta))
                    saw_answer_delta = True
                    sentence_buf += delta
                    while True:
                        cut = _sentence_boundary(sentence_buf)
                        force = len(sentence_buf) > _VOICE_CHUNK_MAX_BUF and cut is not None
                        if cut is None:
                            break
                        candidate_raw = sentence_buf[:cut]
                        strong = _is_strong_flush_boundary(candidate_raw) or force
                        if len(candidate_raw.strip()) < _VOICE_MIN_CHUNK_CHARS and not strong:
                            break
                        sentence_buf = sentence_buf[cut:]
                        chunk = candidate_raw.strip()
                        if not chunk:
                            continue
                        chunk = _strip_chunk_overlap(chunk, last_spoken_tail)
                        if not chunk:
                            continue
                        if os.getenv("VOICE_DEBUG_CHUNK"):
                            logger.debug("VOICE chunk_out=%r rest=%r", chunk, sentence_buf[:120])
                        last_spoken_tail = chunk[-80:] if len(chunk) > 80 else chunk
                        tts_chunk_count += 1
                        if tts_chunk_count == 1:
                            tl.mark(
                                "first_tts_text_yield",
                                chunk_chars=len(chunk),
                                buf_remaining=len(sentence_buf),
                            )
                        tl.mark(
                            "tts_text_yield",
                            chunk_index=tts_chunk_count,
                            chunk_chars=len(chunk),
                            buf_remaining=len(sentence_buf),
                        )
                        yield chunk + " "
                elif otype == "error":
                    err_txt = str(obj.get("errorText") or "unknown error")
                    tl.mark("api_stream_error", error_text=err_txt[:200])
                    logger.warning("query stream error from API: %s", err_txt)
                    apology = (
                        "Maafi, abhi madad nahi ho payi. Dobara boliye."
                        if lang_key == "hi"
                        else "Sorry, I couldn't help right now. Please try again."
                    )
                    tl.finish(
                        outcome="api_error",
                        sse_events=sse_event_count,
                        tts_chunks=tts_chunk_count,
                        fillers=filler_events,
                    )
                    yield apology
                    return
            tl.mark(
                "api_query_stream_end",
                sse_events=sse_event_count,
                saw_answer_delta=saw_answer_delta,
            )
        except Exception:
            tl.mark("api_query_stream_failed")
            logger.exception("Krishi POST %s failed", f"{self._api_base}/api/v1/query/stream")
            tl.finish(outcome="http_exception", sse_events=sse_event_count)
            yield (
                "Sorry, the assistant is temporarily unavailable. Please try again in a moment."
                if lang_key != "hi"
                else "Maafi, assistant abhi uplabdh nahi hai. Thodi der baad koshish karein."
            )
            return

        tail = sentence_buf.strip()
        if tail:
            tail = _strip_chunk_overlap(tail, last_spoken_tail)
            if tail:
                tts_chunk_count += 1
                tl.mark(
                    "tts_text_yield_tail",
                    chunk_chars=len(tail),
                    chunk_index=tts_chunk_count,
                )
                yield tail + " "
        elif not saw_answer_delta:
            tl.mark("no_answer_tokens")
            yield (
                "I could not find an answer. Please try asking in simpler words."
                if lang_key != "hi"
                else "Jawaab nahi mil paya. Seedha sawal dubara boliye."
            )
        tl.finish(
            outcome="ok",
            sse_events=sse_event_count,
            tts_chunks=tts_chunk_count,
            fillers=filler_events,
            answer_received=saw_answer_delta,
        )


async def entrypoint(ctx: JobContext) -> None:
    session_id = f"vsess_{uuid.uuid4().hex[:10]}"
    session_tl = VoiceTimeline(
        turn_id=session_id,
        session_id=session_id,
        room=ctx.room.name,
    )
    session_tl.mark("entrypoint_start")

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    session_tl.mark("livekit_connected")
    _register_lk_agent_request_sink(ctx.room)

    try:
        participant = await ctx.wait_for_participant()
    except RuntimeError as exc:
        # Room disconnected before any participant arrived (e.g. client dropped early).
        # This is a normal transient event — log at INFO and exit cleanly.
        session_tl.mark("participant_wait_failed", error=str(exc)[:200])
        logger.info("Room disconnected before participant arrived, skipping job: %s", exc)
        return
    session_tl.mark("participant_joined", identity=participant.identity)
    meta = _parse_farmer_meta(participant.metadata)

    farmer_id = str(meta.get("farmer_id") or "").strip() or os.getenv(
        "VOICE_DEFAULT_FARMER_ID", ""
    ).strip()
    if not farmer_id:
        logger.error(
            "No farmer_id in participant metadata and VOICE_DEFAULT_FARMER_ID is unset; disconnecting."
        )
        await ctx.room.disconnect()
        return

    raw_conv = meta.get("conversation_id")
    if raw_conv is None or raw_conv == "":
        conv = str(uuid.uuid4())
        logger.info("voice session: no conversation_id in metadata, using %s", conv)
    else:
        conv = str(raw_conv).strip() or str(uuid.uuid4())
    language = str(meta.get("language") or "en").strip() or "en"

    api_base = os.getenv("KRISHI_API_BASE_URL", "http://127.0.0.1:8000").strip()

    dg_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
    if not dg_key:
        logger.error("DEEPGRAM_API_KEY is not set; voice worker needs it for STT and TTS.")
        await ctx.room.disconnect()
        return

    tts_model_name = os.getenv("DEEPGRAM_TTS_MODEL", "aura-2-andromeda-en").strip()

    settings = get_settings()
    twin: FarmerTwin | None = None
    farmer_nm: str | None = None
    session_tl.farmer_id = farmer_id
    session_tl.conversation_id = conv
    session_tl.mark("resolve_farmer_twin_start")
    try:
        twin = await asyncio.wait_for(
            resolve_farmer_twin(farmer_id, "online", settings),
            timeout=2.0,
        )
        if twin and twin.name.strip():
            farmer_nm = twin.name.strip()
        session_tl.mark("resolve_farmer_twin_done", twin_found=twin is not None)
    except asyncio.TimeoutError:
        session_tl.mark("resolve_farmer_twin_timeout", timeout_s=2.0)
    except Exception:
        session_tl.mark("resolve_farmer_twin_failed")
        logger.debug("resolve_farmer_twin for voice session failed", exc_info=True)

    session_tl.mark("vad_load_start")
    vad = silero.VAD.load()
    session_tl.mark("vad_load_done")
    dg_lang = _deepgram_stt_language(language)
    session_tl.mark("deepgram_stt_init", stt_language=dg_lang)
    stt_model = deepgram.STT(
        model="nova-3",
        language=dg_lang,
        detect_language=False,
        api_key=dg_key,
    )
    session_tl.mark("deepgram_tts_init", tts_model=tts_model_name)
    tts_model = deepgram.TTS(api_key=dg_key, model=tts_model_name)

    session_tl.mark(
        "agent_session_create",
        endpointing_min_s=0.4,
        endpointing_max_s=3.0,
    )
    stream_client = httpx.AsyncClient(
        timeout=120.0,
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
    )
    session = AgentSession(
        vad=vad,
        stt=stt_model,
        tts=tts_model,
        turn_handling={
            "endpointing": {"min_delay": 0.4, "max_delay": 3.0},
            "interruption": {"enabled": True, "mode": "vad"},
        },
    )
    agent = KrishiVoiceAgent(
        api_base=api_base,
        farmer_id=farmer_id,
        conversation_id=conv,
        language=language,
        http_client=stream_client,
        farmer_name=farmer_nm,
        session_id=session_id,
        room_name=ctx.room.name,
    )
    room_opts = room_io.RoomOptions(participant_identity=participant.identity)
    session_tl.mark("agent_session_start")
    try:
        await session.start(agent, room=ctx.room, room_options=room_opts)
    except RuntimeError as exc:
        session_tl.mark("agent_session_start_failed", error=str(exc)[:200])
        logger.info("Session could not start (room already gone?): %s", exc)
        await stream_client.aclose()
        return
    session_tl.mark("agent_session_ready")

    welcome_key = f"{ctx.room.name}:{farmer_id}"
    if welcome_key not in _WELCOME_SENT_KEYS:
        welcome_text = _build_voice_welcome(twin, language)
        session_tl.mark("welcome_tts_start", welcome_chars=len(welcome_text))
        try:
            handle = session.say(welcome_text, add_to_chat_ctx=False)
            await handle.wait_for_playout()
            session_tl.mark("welcome_tts_done")
        except Exception:
            session_tl.mark("welcome_tts_failed")
            logger.warning("voice welcome TTS failed", exc_info=True)
        _WELCOME_SENT_KEYS.add(welcome_key)
    else:
        session_tl.mark("welcome_tts_skipped", reason="already_sent")

    await stream_client.aclose()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="krishi-voice-agent",
            # Default prod threshold is 0.7; a single voice job often reports ~0.72 load,
            # which flips the worker unavailable and breaks dispatch. 1.0 is valid for prod.
            load_threshold=1.0,
        )
    )
