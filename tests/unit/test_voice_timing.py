"""Voice timeline logging helpers."""

from __future__ import annotations

import logging

import pytest

from voice_agent.timing import VoiceTimeline, voice_timing_enabled


def test_voice_timing_enabled_default(monkeypatch):
    monkeypatch.delenv("VOICE_TIMING_LOG", raising=False)
    assert voice_timing_enabled() is True


def test_voice_timeline_emits_structured_log(monkeypatch, caplog):
    monkeypatch.setenv("VOICE_TIMING_LOG", "true")
    caplog.set_level(logging.INFO, logger="krishi.voice")
    tl = VoiceTimeline(turn_id="vturn_test", farmer_id="f1", conversation_id="c1")
    tl.mark("unit_test_phase", foo="bar")
    assert any(
        r.name == "krishi.voice" and "voice_phase" in r.getMessage()
        for r in caplog.records
    )
    assert tl.phases[0][0] == "unit_test_phase"


def test_voice_timeline_respects_disable(monkeypatch, caplog):
    monkeypatch.setenv("VOICE_TIMING_LOG", "false")
    caplog.set_level(logging.INFO, logger="krishi.voice")
    tl = VoiceTimeline(turn_id="vturn_off")
    tl.mark("should_not_log")
    assert not [r for r in caplog.records if r.name == "krishi.voice"]
