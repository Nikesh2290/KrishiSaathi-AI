"""Voice token endpoint (LiveKit handshake)."""

from __future__ import annotations

import json
import uuid

import jwt
import pytest
from fastapi.testclient import TestClient

from config.settings import get_settings


class _NoLiveKitSettings:
    livekit_configured = False


@pytest.fixture
def voice_client_no_livekit(monkeypatch):
    monkeypatch.setattr("api.routes.voice.get_settings", lambda: _NoLiveKitSettings())
    from api.main import create_app

    return TestClient(create_app())


def test_voice_token_returns_503_when_livekit_unconfigured(voice_client_no_livekit):
    r = voice_client_no_livekit.post(
        "/api/v1/voice/token",
        json={"farmer_id": str(uuid.uuid4())},
    )
    assert r.status_code == 503


class _FakeAgentDispatch:
    async def create_dispatch(self, _req):
        return None


class _FakeLiveKitAPI:
    def __init__(self, *args, **kwargs):
        self.agent_dispatch = _FakeAgentDispatch()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


def test_voice_token_mints_jwt(monkeypatch):
    monkeypatch.setenv("LIVEKIT_URL", "wss://example.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "testkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "0123456789abcdef0123456789abcdef")
    monkeypatch.setattr("api.routes.voice.LiveKitAPI", _FakeLiveKitAPI)
    get_settings.cache_clear()

    from api.main import create_app

    client = TestClient(create_app())
    farmer_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    r = client.post(
        "/api/v1/voice/token",
        json={"farmer_id": farmer_id, "conversation_id": None, "language": "hi"},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["server_url"] == "wss://example.livekit.cloud"
    assert data["participant_identity"].startswith("farmer-")
    assert data["room_name"].startswith("krishi-")
    tok = data["participant_token"]
    assert isinstance(tok, str) and len(tok) > 20

    claims = jwt.decode(tok, options={"verify_signature": False})
    assert claims["iss"] == "testkey"
    assert claims["sub"] == data["participant_identity"]
    assert claims["video"]["room"] == data["room_name"]
    assert claims["video"]["roomJoin"] is True
    m = json.loads(claims["metadata"])
    assert m["farmer_id"] == farmer_id
