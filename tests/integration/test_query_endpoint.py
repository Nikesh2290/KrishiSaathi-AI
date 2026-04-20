"""API integration tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from models.response import AgentResponse


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_query_mocked(monkeypatch, client):
    async def fake_run(req):
        return AgentResponse(text="ok", tool_trace=["scheme"])

    monkeypatch.setattr("api.routes.query.run", fake_run)
    body = {
        "farmer_id": "f1",
        "query": {"text": "PM-KISAN", "language": "hi"},
        "context": {"connectivity": "online", "device_intent": "scheme_query", "location": {}},
    }
    r = client.post("/api/v1/query", json=body)
    assert r.status_code == 200
    assert r.json()["text"] == "ok"


def test_stream_emits(monkeypatch, client):
    async def fake_stream(_req):
        yield {"step": "route", "status": "done"}
        yield {"step": "plan", "status": "done", "tools": ["climate"]}

    monkeypatch.setattr("api.routes.query.stream_run", fake_stream)
    r = client.get(
        "/api/v1/query/stream",
        params={"farmer_id": "f1", "q": "weather", "connectivity": "online"},
    )
    assert r.status_code == 200
    assert b"data:" in r.content
