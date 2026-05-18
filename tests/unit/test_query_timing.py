"""Query timeline logging helpers."""

from __future__ import annotations

import logging

import pytest

from agent.timing import QueryTimeline, query_timing_enabled
from config.logging import request_id_var


def test_query_timing_enabled_default(monkeypatch):
    monkeypatch.delenv("QUERY_TIMING_LOG", raising=False)
    monkeypatch.delenv("VOICE_TIMING_LOG", raising=False)
    assert query_timing_enabled() is True


def test_query_timeline_mark_with_logging_record_factory(monkeypatch, caplog):
    """Regression: extra request_id must not clash with record_factory injection."""
    monkeypatch.setenv("QUERY_TIMING_LOG", "true")

    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.request_id = request_id_var.get()
        return record

    logging.setLogRecordFactory(record_factory)

    token = request_id_var.set("rid_test_123")
    try:
        caplog.set_level(logging.INFO, logger="krishi.query")
        tl = QueryTimeline(request_id="rid_test_123", farmer_id="f1", device_intent="voice")
        tl.mark("unit_test_phase", foo="bar")
    finally:
        request_id_var.reset(token)
        logging.setLogRecordFactory(old_factory)

    records = [r for r in caplog.records if r.name == "krishi.query"]
    assert records, "expected query_phase log line"
    assert records[0].request_id == "rid_test_123"
    assert getattr(records[0], "phase", None) == "unit_test_phase"


def test_query_timeline_respects_disable(monkeypatch, caplog):
    monkeypatch.setenv("QUERY_TIMING_LOG", "false")
    monkeypatch.setenv("VOICE_TIMING_LOG", "false")
    caplog.set_level(logging.INFO, logger="krishi.query")
    tl = QueryTimeline(request_id="rid_off")
    tl.mark("should_not_log")
    assert not [r for r in caplog.records if r.name == "krishi.query"]
