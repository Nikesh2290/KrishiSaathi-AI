"""Structured latency logging for voice-to-voice turns (grep: voice_phase)."""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

import logging

logger = logging.getLogger("krishi.voice")


def voice_timing_enabled() -> bool:
    return os.getenv("VOICE_TIMING_LOG", "true").strip().lower() in ("1", "true", "yes")


def new_turn_id() -> str:
    return f"vturn_{uuid.uuid4().hex[:12]}"


class VoiceTimeline:
    """Per-turn or per-session wall-clock marks (ms since construction)."""

    def __init__(
        self,
        *,
        turn_id: str | None = None,
        session_id: str | None = None,
        farmer_id: str = "",
        conversation_id: str | None = None,
        room: str | None = None,
    ) -> None:
        self.turn_id = turn_id or new_turn_id()
        self.session_id = session_id
        self.farmer_id = farmer_id
        self.conversation_id = conversation_id
        self.room = room
        self._t0 = time.perf_counter()
        self.phases: list[tuple[str, float]] = []

    def _base_extra(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "turn_id": self.turn_id,
            "farmer_id": self.farmer_id or None,
            "conversation_id": self.conversation_id,
        }
        if self.session_id:
            out["session_id"] = self.session_id
        if self.room:
            out["room"] = self.room
        return out

    def mark(self, phase: str, **extra: Any) -> float:
        ms = (time.perf_counter() - self._t0) * 1000.0
        self.phases.append((phase, ms))
        if voice_timing_enabled():
            payload = {**self._base_extra(), "phase": phase, "ms_since_start": round(ms, 2)}
            payload.update(extra)
            logger.info("voice_phase", extra=payload)
        return ms

    def finish(self, **extra: Any) -> None:
        self.mark("turn_complete", **extra)
