"""Outbound API payloads."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


ConfidenceLevel = Literal["high", "medium", "low"]
DataSource = Literal["live", "offline"]
FallbackHint = Literal["USE_ONDEVICE", "RETRY_ONLINE_LATER"]


class StructuredResult(BaseModel):
    kind: str = "general"
    data: Dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    response_id: str = Field(default_factory=lambda: str(uuid4()))
    conversation_id: Optional[str] = None
    text: str = ""
    structured: StructuredResult = Field(default_factory=StructuredResult)
    data_source: DataSource = "live"
    confidence_level: ConfidenceLevel = "medium"
    confidence_score: float = 0.5
    model_used: str = ""
    tool_trace: List[str] = Field(default_factory=list)
    safety_flags: List[str] = Field(default_factory=list)
    fallback_hint: Optional[FallbackHint] = None
    language: str = "en"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def model_dump_json_safe(self) -> dict:
        return self.model_dump(mode="json")
