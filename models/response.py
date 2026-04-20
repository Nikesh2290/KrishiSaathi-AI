"""Outbound API payloads."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


ConfidenceLevel = Literal["high", "medium", "low"]
DataSource = Literal["live", "offline"]


class StructuredResult(BaseModel):
    """Flexible structured block per intent (disease, weather, schemes, etc.)."""

    kind: str = "general"
    data: Dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    response_id: str = Field(default_factory=lambda: str(uuid4()))
    text: str = ""
    structured: StructuredResult = Field(default_factory=StructuredResult)
    data_source: DataSource = "live"
    confidence_level: ConfidenceLevel = "medium"
    tool_trace: List[str] = Field(default_factory=list)
    language: str = "hi"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    safety_flags: List[str] = Field(default_factory=list)

    def model_dump_json_safe(self) -> dict:
        d = self.model_dump(mode="json")
        return d
