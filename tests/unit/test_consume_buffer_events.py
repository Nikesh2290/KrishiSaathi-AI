"""SSE buffer parsing for streaming voice client."""

from __future__ import annotations

from voice_agent.query_stream_client import _consume_buffer_events


def test_consume_buffer_emits_multiple_frames():
    buf = (
        'data: {"type":"data-tool","data":{"tool":"routing","status":"started"}}\n\n'
        'data: {"type":"text-delta","id":"t","delta":"Hi"}\n\n'
    )
    rest, status, objs = _consume_buffer_events(buf)
    assert rest == ""
    assert status == ""
    assert objs[0]["type"] == "data-tool"
    assert objs[1]["type"] == "text-delta"


def test_consume_buffer_done_breaks():
    buf = 'data: {"type":"x"}\n\ndata: [DONE]\n\n'
    rest, status, objs = _consume_buffer_events(buf)
    assert status == "done"
    assert len(objs) == 1
