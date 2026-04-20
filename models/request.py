"""Inbound API payloads."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class QueryPayload(BaseModel):
    text: str = ""
    voice_b64: Optional[str] = None
    image_b64: Optional[str] = None
    language: str = "hi"


class ContextPayload(BaseModel):
    location: Dict[str, Any] = Field(default_factory=dict)
    connectivity: str = "online"  # online | offline
    device_intent: str = "general"


class AgentRequest(BaseModel):
    farmer_id: str
    query: QueryPayload = Field(default_factory=QueryPayload)
    context: ContextPayload = Field(default_factory=ContextPayload)
