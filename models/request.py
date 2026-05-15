"""Inbound API payloads."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class QueryPayload(BaseModel):
    text: str = ""
    voice_b64: Optional[str] = None
    image_ref: Optional[str] = None
    language: str = "en"


class ContextPayload(BaseModel):
    location: Dict[str, Any] = Field(default_factory=dict)
    connectivity: str = "online"
    device_intent: str = "general"
    device_capabilities: Dict[str, Any] = Field(default_factory=dict)


class AgentRequest(BaseModel):
    farmer_id: str
    conversation_id: Optional[str] = None
    query: QueryPayload = Field(default_factory=QueryPayload)
    context: ContextPayload = Field(default_factory=ContextPayload)
