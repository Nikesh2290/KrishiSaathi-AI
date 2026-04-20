"""API integration tests."""

from __future__ import annotations

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

    async def fake_log_query(*_args, **_kwargs):
        return None

    monkeypatch.setattr("api.routes.query.run_graph", fake_run_graph)
    monkeypatch.setattr("api.routes.query.log_query", fake_log_query)
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
