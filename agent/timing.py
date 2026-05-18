"""Backend query phase timing (grep: query_phase). Used heavily for device_intent=voice."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger("krishi.query")


def query_timing_enabled() -> bool:
    if os.getenv("QUERY_TIMING_LOG", "").strip():
        return os.getenv("QUERY_TIMING_LOG", "").strip().lower() in ("1", "true", "yes")
    # Default on for voice so mobile/voice worker diagnosis works without extra env.
    return os.getenv("VOICE_TIMING_LOG", "true").strip().lower() in ("1", "true", "yes")


class QueryTimeline:
    def __init__(
        self,
        *,
        request_id: str | None = None,
        voice_turn_id: str | None = None,
        farmer_id: str = "",
        conversation_id: str | None = None,
        device_intent: str = "",
    ) -> None:
        self.request_id = request_id
        self.voice_turn_id = voice_turn_id
        self.farmer_id = farmer_id
        self.conversation_id = conversation_id
        self.device_intent = device_intent
        self._t0 = time.perf_counter()

    def _base_extra(self) -> dict[str, Any]:
        # request_id is injected on every LogRecord by config.logging record_factory;
        # do not pass it in extra= or logging raises KeyError on overwrite.
        out: dict[str, Any] = {
            "farmer_id": self.farmer_id or None,
            "conversation_id": self.conversation_id,
            "device_intent": self.device_intent or None,
        }
        if self.voice_turn_id:
            out["voice_turn_id"] = self.voice_turn_id
        return out

    def mark(self, phase: str, **extra: Any) -> float:
        ms = (time.perf_counter() - self._t0) * 1000.0
        if query_timing_enabled():
            payload = {**self._base_extra(), "phase": phase, "ms_since_start": round(ms, 2)}
            payload.update(extra)
            logger.info("query_phase", extra=payload)
        return ms

    def finish(self, **extra: Any) -> None:
        self.mark("query_complete", **extra)
