"""API integration tests."""

from __future__ import annotations

import json

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


def test_query_mocked(monkeypatch, client):
    async def fake_run_graph(_req):
        return {
            "draft_text": "ok",
            "tool_results": {"scheme_0": {"answer": "x"}},
            "tool_trace": ["scheme"],
            "data_source": "live",
            "safety_flags": [],
            "model_used": "mock-model",
            "confidence_score": 0.75,
            "fallback_hint": None,
        }

    async def fake_persist_log_query(*_args, **_kwargs):
        return None

    monkeypatch.setattr("api.routes.query.run_graph", fake_run_graph)
    monkeypatch.setattr("api.routes.query.persist_log_query", fake_persist_log_query)
    body = {
        "farmer_id": "f1",
        "query": {"text": "PM-KISAN", "language": "hi"},
        "context": {"connectivity": "online", "device_intent": "scheme_query", "location": {}},
    }
    r = client.post("/api/v1/query", json=body)
    assert r.status_code == 200
    payload = r.json()
    assert payload["text"] == "ok"
    assert payload["structured"]["kind"] == "scheme"
    assert payload["model_used"] == "mock-model"
    assert 0.0 <= payload["confidence_score"] <= 1.0
    assert payload["fallback_hint"] in (None, "USE_ONDEVICE", "RETRY_ONLINE_LATER")


def test_query_stream_mocked(monkeypatch, client):
    """POST /query/stream uses AI SDK Data Stream (data: lines + [DONE])."""

    async def fake_stream(_req):
        yield ("start", {"messageId": "00000000-0000-0000-0000-000000000099"})
        yield ("data-stage", {"data": {"stage": "routing"}})
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
    assert payloads[1] == {
        "type": "data-stage",
        "data": {"stage": "routing"},
    }
    assert payloads[2] == {"type": "start-step"}
    assert payloads[3] == {"type": "text-start", "id": "txt_test"}
    assert payloads[4] == {"type": "text-delta", "id": "txt_test", "delta": "hello"}
    assert payloads[5] == {"type": "text-end", "id": "txt_test"}
    assert payloads[6] == {"type": "finish-step"}
    assert payloads[7]["type"] == "data-metadata"
    assert payloads[7]["data"]["confidence_score"] == 0.75
    assert payloads[7]["data"]["structured"]["kind"] == "scheme"
    assert "text" not in payloads[7]["data"]
    assert payloads[8] == {"type": "finish"}
    assert payloads[-1] == "[DONE]"
