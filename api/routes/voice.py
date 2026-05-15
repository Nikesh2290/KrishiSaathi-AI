"""LiveKit voice room token (handshake) — no secrets returned except short-lived JWT."""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import timedelta

from fastapi import APIRouter, HTTPException
from livekit.api import AccessToken, CreateAgentDispatchRequest, LiveKitAPI, VideoGrants

from config.settings import get_settings
from db.persistence import resolve_farmer_twin
from models.voice import VoiceTokenRequest, VoiceTokenResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/voice", tags=["voice"])

_ROOM_SAFE = re.compile(r"[^\w\-]")


def _sanitize_room_fragment(name: str) -> str:
    s = _ROOM_SAFE.sub("-", name.strip())[:48]
    return s or "room"


@router.post("/token", response_model=VoiceTokenResponse)
async def post_voice_token(body: VoiceTokenRequest) -> VoiceTokenResponse:
    settings = get_settings()
    if not settings.livekit_configured:
        raise HTTPException(
            status_code=503,
            detail="LiveKit is not configured (set LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET).",
        )

    room = body.room_name or f"krishi-{_sanitize_room_fragment(body.farmer_id)}-{uuid.uuid4().hex[:10]}"
    identity = body.participant_identity or f"farmer-{body.farmer_id[:8]}"

    display_name = "Farmer"
    try:
        twin_v = await resolve_farmer_twin(body.farmer_id, "online", settings)
        if twin_v and twin_v.name.strip():
            display_name = twin_v.name.strip()[:64]
    except Exception as exc:
        logger.debug("resolve_farmer_twin for voice display name: %s", exc)

    meta = {
        "farmer_id": body.farmer_id,
        "conversation_id": body.conversation_id,
        "language": body.language or "hi",
    }
    try:
        token = (
            AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
            .with_identity(identity)
            .with_name(display_name)
            .with_metadata(json.dumps(meta, ensure_ascii=False))
            .with_ttl(timedelta(seconds=int(settings.voice_token_ttl_seconds)))
            .with_grants(
                VideoGrants(
                    room_join=True,
                    room=room,
                    can_publish=True,
                    can_subscribe=True,
                    can_publish_data=True,
                )
            )
            .to_jwt()
        )
    except ValueError as e:
        logger.warning("livekit token build failed: %s", e)
        raise HTTPException(status_code=500, detail="Could not mint voice token.") from e

    try:
        async with LiveKitAPI(
            url=settings.livekit_url,
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret,
        ) as lkapi:
            await lkapi.agent_dispatch.create_dispatch(
                CreateAgentDispatchRequest(
                    agent_name="krishi-voice-agent",
                    room=room,
                )
            )
        logger.info("agent dispatch created for room %s", room)
    except Exception as exc:
        logger.warning("agent dispatch failed for room %s: %s", room, exc)

    return VoiceTokenResponse(
        server_url=settings.livekit_url,
        room_name=room,
        participant_token=token,
        participant_identity=identity,
    )
