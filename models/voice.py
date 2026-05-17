"""LiveKit voice handshake payloads."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class VoiceTokenRequest(BaseModel):
    farmer_id: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None
    room_name: Optional[str] = None
    participant_identity: Optional[str] = None
    language: str = Field(default="en", min_length=1, max_length=16)


class VoiceTokenResponse(BaseModel):
    server_url: str
    room_name: str
    participant_token: str
    participant_identity: str
    conversation_id: str = Field(
        ...,
        description="Thread id for voice/text memory; minted when the client omits one.",
    )
