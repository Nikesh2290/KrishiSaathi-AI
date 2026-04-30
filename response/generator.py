"""Build structured AgentResponse from graph output + safety."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from models.response import AgentResponse, ConfidenceLevel, FallbackHint, StructuredResult


def _intent_from_trace(trace: List[str]) -> str:
    if "clarify" in trace:
        return "clarify"
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


def _level_from_score(score: float) -> ConfidenceLevel:
    if score >= 0.80:
        return "high"
    if score >= 0.60:
        return "medium"
    return "low"


def build(
    draft_text: str,
    tool_results: Dict[str, Any],
    tool_trace: List[str],
    data_source: str,
    language: str,
    safety_flags: List[str],
    model_used: str = "",
    confidence_score: float = 0.5,
    fallback_hint: Optional[FallbackHint] = None,
) -> AgentResponse:
    intent = _intent_from_trace(tool_trace)
    if "low_confidence_vision" in safety_flags:
        confidence_score = min(confidence_score, 0.4)

    structured_data: Dict[str, Any] = {"intent": intent, "tool_results": tool_results}
    if intent == "disease":
        for v in tool_results.values():
            if not isinstance(v, dict):
                continue
            if "is_agricultural" in v:
                if v.get("is_agricultural") is False:
                    intent = "image_general"
                structured_data.update(v)
                break
            if v.get("disease"):
                structured_data.update(v)
                break

    structured_data["intent"] = intent

    return AgentResponse(
        text=draft_text,
        structured=StructuredResult(kind=intent, data=structured_data),
        data_source="offline" if data_source == "offline" else "live",  # type: ignore[arg-type]
        confidence_level=_level_from_score(confidence_score),
        confidence_score=round(float(confidence_score), 3),
        model_used=model_used,
        tool_trace=tool_trace,
        language=language,
        safety_flags=safety_flags,
        fallback_hint=fallback_hint,
        timestamp=datetime.now(timezone.utc),
    )
