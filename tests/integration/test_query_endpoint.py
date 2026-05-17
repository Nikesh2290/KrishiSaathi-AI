"""API integration tests."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_query_stream_mocked(monkeypatch, client):
    """POST /query/stream uses AI SDK Data Stream (data: lines + [DONE])."""

    async def fake_stream(_req):
        yield ("start", {"messageId": "00000000-0000-0000-0000-000000000099"})
        yield ("start-step", {})
        yield ("text-start", {"id": "txt_test"})
        yield ("text-delta", {"id": "txt_test", "delta": "hello"})
        yield ("text-end", {"id": "txt_test"})
        yield ("finish-step", {})
        meta = {
            "response_id": "00000000-0000-0000-0000-000000000001",
            "structured": {"kind": "scheme", "data": {}},
            "data_source": "live",
            "confidence_level": "medium",
            "confidence_score": 0.75,
            "model_used": "mock",
            "tool_trace": ["scheme"],
            "safety_flags": [],
            "fallback_hint": None,
            "language": "hi",
            "timestamp": "2026-01-01T00:00:00Z",
        }
        yield ("data-metadata", {"data": meta})
        yield ("finish", {})
        yield ("__done__", None)

    monkeypatch.setattr("api.routes.query.run_graph_stream", fake_stream)
    body = {
        "farmer_id": "f1",
        "query": {"text": "PM-KISAN", "language": "hi"},
        "context": {"connectivity": "online", "device_intent": "scheme_query", "location": {}},
    }
    with client.stream(
        "POST",
        "/api/v1/query/stream",
        json=body,
        headers={"Accept": "text/event-stream"},
    ) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "").lower()
        assert r.headers.get("x-vercel-ai-ui-message-stream") == "v1"

        payloads: list[object] = []
        buffer = ""

        for line in r.iter_lines():
            if line is None:
                continue
            buffer += line + "\n"
            while "\n\n" in buffer:
                block, buffer = buffer.split("\n\n", 1)
                data_lines: list[str] = []
                for part in block.strip().split("\n"):
                    if part.startswith("data:"):
                        data_lines.append(part.removeprefix("data:").strip())
                data_str = "".join(data_lines).strip()
                if data_str == "[DONE]":
                    payloads.append("[DONE]")
                else:
                    payloads.append(json.loads(data_str))

    assert payloads[0] == {
        "type": "start",
        "messageId": "00000000-0000-0000-0000-000000000099",
    }
    assert payloads[1] == {"type": "start-step"}
    assert payloads[2] == {"type": "text-start", "id": "txt_test"}
    assert payloads[3] == {"type": "text-delta", "id": "txt_test", "delta": "hello"}
    assert payloads[4] == {"type": "text-end", "id": "txt_test"}
    assert payloads[5] == {"type": "finish-step"}
    assert payloads[6]["type"] == "data-metadata"
    assert payloads[6]["data"]["confidence_score"] == 0.75
    assert payloads[6]["data"]["structured"]["kind"] == "scheme"
    assert "text" not in payloads[6]["data"]
    assert payloads[7] == {"type": "finish"}
    assert payloads[-1] == "[DONE]"


def _sse_payloads(response_iter_lines):
    payloads: list[object] = []
    buffer = ""
    for line in response_iter_lines:
        if line is None:
            continue
        buffer += line + "\n"
        while "\n\n" in buffer:
            block, buffer = buffer.split("\n\n", 1)
            data_lines: list[str] = []
            for part in block.strip().split("\n"):
                if part.startswith("data:"):
                    data_lines.append(part.removeprefix("data:").strip())
            data_str = "".join(data_lines).strip()
            if data_str == "[DONE]":
                payloads.append("[DONE]")
            else:
                payloads.append(json.loads(data_str))
    return payloads


def test_smalltalk_sse_personalized_and_tool_events(client, monkeypatch):
    """Tier 1: farmer twin name + data-tool smalltalk frames."""
    from models.farmer import FarmerTwin

    monkeypatch.setattr(
        "api.routes.query.resolve_farmer_twin",
        AsyncMock(return_value=FarmerTwin(farmer_id="f1", name="Ramesh")),
    )
    body = {
        "farmer_id": "f1",
        "query": {"text": "hello", "language": "en"},
        "context": {"connectivity": "online", "device_intent": "general", "location": {}},
    }
    with client.stream(
        "POST",
        "/api/v1/query/stream",
        json=body,
        headers={"Accept": "text/event-stream"},
    ) as r:
        assert r.status_code == 200
        payloads = _sse_payloads(r.iter_lines())

    tools_smalltalk = [p for p in payloads if isinstance(p, dict) and p.get("type") == "data-tool" and p.get("data", {}).get("tool") == "smalltalk"]
    assert tools_smalltalk and tools_smalltalk[0]["data"]["status"] == "started"

    deltas = [
        p.get("delta", "")
        for p in payloads
        if isinstance(p, dict) and p.get("type") == "text-delta"
    ]
    full_text = "".join(deltas)
    assert "Ramesh" in full_text


def test_direct_llm_stream_parallel(monkeypatch, client):
    """Tier 2: general query skips planner tools path when mocked LLM streams."""

    async def fake_gs(_messages, prefer_local=False, settings=None):
        yield "Paris "
        yield "is the capital."

    monkeypatch.setattr("agent.graph.generate_stream", fake_gs)

    body = {
        "farmer_id": "f1",
        "query": {"text": "capital of France in one line", "language": "en"},
        "context": {"connectivity": "online", "device_intent": "general", "location": {}},
    }
    with client.stream(
        "POST",
        "/api/v1/query/stream",
        json=body,
        headers={"Accept": "text/event-stream"},
    ) as r:
        assert r.status_code == 200
        payloads = _sse_payloads(r.iter_lines())

    meta = next(
        p
        for p in payloads
        if isinstance(p, dict) and p.get("type") == "data-metadata"
    )
    assert meta["data"]["tool_trace"] == ["direct_llm"]

    thinking_evts = [
        p
        for p in payloads
        if isinstance(p, dict)
        and p.get("type") == "data-tool"
        and p.get("data", {}).get("tool") == "thinking"
    ]
    assert not thinking_evts

    deltas = "".join(
        p.get("delta", "")
        for p in payloads
        if isinstance(p, dict) and p.get("type") == "text-delta"
    )
    assert "Paris" in deltas
