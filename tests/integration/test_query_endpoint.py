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
    """SSE /query/stream yields status → token → done with AgentResponse-shaped JSON."""

    async def fake_stream(_req):
        yield ("status", json.dumps({"stage": "routing"}))
        yield ("token", json.dumps({"text": "hello"}))
        done_payload = {
            "response_id": "00000000-0000-0000-0000-000000000001",
            "text": "hello",
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
        yield ("done", json.dumps(done_payload))

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

        chunks: list[dict] = []
        buffer = ""

        # TestClient exposes .iter_lines() on streaming responses (starlette httpx wrapper).
        for line in r.iter_lines():
            if line is None:
                continue
            buffer += line + "\n"
            # One SSE event ends with blank line (\n\n in buffer).
            while "\n\n" in buffer:
                block, buffer = buffer.split("\n\n", 1)
                event_name = None
                data_lines: list[str] = []
                for part in block.strip().split("\n"):
                    if part.startswith("event:"):
                        event_name = part.removeprefix("event:").strip()
                    elif part.startswith("data:"):
                        data_lines.append(part.removeprefix("data:").strip())
                assert event_name is not None
                data_str = "".join(data_lines)
                chunks.append({"event": event_name, "data": data_str})

    assert chunks[0]["event"] == "status"
    assert json.loads(chunks[0]["data"]) == {"stage": "routing"}
    assert chunks[1]["event"] == "token"
    assert json.loads(chunks[1]["data"]) == {"text": "hello"}
    assert chunks[-1]["event"] == "done"
    finished = json.loads(chunks[-1]["data"])
    assert finished["text"] == "hello"
    assert finished["structured"]["kind"] == "scheme"
