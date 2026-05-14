"""LiveKit Agents entrypoint: STT + HTTP Krishi /query/stream + TTS."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
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

from voice_agent.query_stream_client import collect_text_from_query_stream
from voice_agent.stub_llm import KrishiStubLLM

# LiveKit CLI reads os.environ before connecting; load repo .env regardless of cwd.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

logger = logging.getLogger(__name__)

# LiveKit may send control text streams on this topic; drain them so the RTC layer
# does not log "no callback attached" and streams complete cleanly.
_LK_AGENT_REQUEST_TOPIC = "lk.agent.request"


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
    raw = (locale or "hi").strip().replace("_", "-").lower()
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


class KrishiVoiceAgent(Agent):
    def __init__(
        self,
        *,
        api_base: str,
        farmer_id: str,
        conversation_id: Optional[str],
        language: str,
    ) -> None:
        self._api_base = api_base.rstrip("/")
        self._farmer_id = farmer_id
        self._conversation_id = conversation_id
        self._language = language or "hi"
        super().__init__(
            instructions=(
                "You are Krishi Saathi, an Indian agriculture assistant. "
                "Answers are produced by the Krishi backend from the farmer's speech."
            ),
            llm=KrishiStubLLM(),
        )

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ):
        del tools, model_settings
        user_text = _last_user_message_text(chat_ctx)
        if not user_text:
            return "Please say your farming question again."

        payload: dict[str, Any] = {
            "farmer_id": self._farmer_id,
            "conversation_id": self._conversation_id,
            "query": {"text": user_text, "language": self._language},
            "context": {
                "connectivity": "online",
                "location": {},
                "device_intent": "voice",
                "device_capabilities": {},
            },
        }
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                text = await collect_text_from_query_stream(
                    client,
                    api_base=self._api_base,
                    payload=payload,
                )
        except Exception:
            logger.exception("Krishi POST %s failed", f"{self._api_base}/api/v1/query/stream")
            return "Sorry, the assistant is temporarily unavailable. Please try again in a moment."
        if not text:
            return "I could not find an answer. Please try asking in simpler words."
        return text


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    _register_lk_agent_request_sink(ctx.room)

    try:
        participant = await ctx.wait_for_participant()
    except RuntimeError as exc:
        # Room disconnected before any participant arrived (e.g. client dropped early).
        # This is a normal transient event — log at INFO and exit cleanly.
        logger.info("Room disconnected before participant arrived, skipping job: %s", exc)
        return
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
        conv = None
    else:
        conv = str(raw_conv).strip() or None
    language = str(meta.get("language") or "hi").strip() or "hi"

    api_base = os.getenv("KRISHI_API_BASE_URL", "http://127.0.0.1:8000").strip()

    dg_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
    if not dg_key:
        logger.error("DEEPGRAM_API_KEY is not set; voice worker needs it for STT and TTS.")
        await ctx.room.disconnect()
        return

    tts_model_name = os.getenv("DEEPGRAM_TTS_MODEL", "aura-2-andromeda-en").strip()

    vad = silero.VAD.load()
    dg_lang = _deepgram_stt_language(language)
    stt_model = deepgram.STT(
        model="nova-3",
        language=dg_lang,
        detect_language=False,
        api_key=dg_key,
    )
    tts_model = deepgram.TTS(api_key=dg_key, model=tts_model_name)

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
    )
    room_opts = room_io.RoomOptions(participant_identity=participant.identity)
    try:
        await session.start(agent, room=ctx.room, room_options=room_opts)
    except RuntimeError as exc:
        logger.info("Session could not start (room already gone?): %s", exc)
        return


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
