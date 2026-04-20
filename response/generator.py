"""Build structured AgentResponse from graph output + safety."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from models.response import AgentResponse, ConfidenceLevel, StructuredResult


def _confidence_from_tools(tool_results: Dict[str, Any]) -> ConfidenceLevel:
    for v in tool_results.values():
        if isinstance(v, dict) and float(v.get("confidence") or 1.0) < 0.5:
            return "low"
    return "medium"


def _intent_from_trace(trace: List[str]) -> str:
    if "vision" in trace:
        return "disease"
    if "climate" in trace:
        return "weather"
    if "scheme" in trace:
        return "scheme"
    if "market" in trace:
        return "market"
    if "crop_planner" in trace:
        return "crop_plan"
    if "financial" in trace:
        return "financial"
    return "general"


def build(
    draft_text: str,
    tool_results: Dict[str, Any],
    tool_trace: List[str],
    data_source: str,
    language: str,
    safety_flags: List[str],
) -> AgentResponse:
    intent = _intent_from_trace(tool_trace)
    conf: ConfidenceLevel = _confidence_from_tools(tool_results)
    if "low_confidence_vision" in safety_flags:
        conf = "low"

    structured_data: Dict[str, Any] = {"intent": intent, "tool_results": tool_results}
    if intent == "disease":
        for v in tool_results.values():
            if isinstance(v, dict) and v.get("disease"):
                structured_data.update(v)
                break

    return AgentResponse(
        text=draft_text,
        structured=StructuredResult(kind=intent, data=structured_data),
        data_source="offline" if data_source == "offline" else "live",  # type: ignore[arg-type]
        confidence_level=conf,
        tool_trace=tool_trace,
        language=language,
        timestamp=datetime.now(timezone.utc),
        safety_flags=safety_flags,
    )
