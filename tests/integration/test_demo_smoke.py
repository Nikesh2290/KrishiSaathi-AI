"""Must-pass-before-submission end-to-end smoke test.

Exercises the four demo beats. Failure here = do not submit.
"""

from __future__ import annotations

import gzip
import json

import pytest
from httpx import ASGITransport, AsyncClient

from tests.fakes.fake_gemma_client import install_fake


@pytest.mark.asyncio
async def test_demo_smoke_all_four_beats(monkeypatch):
    # Import the app first so agent.graph and modules.vision resolve their
    # `generate` / `generate_with_vision` bindings against the real module
    # attributes. install_fake() then swaps those attributes so both the
    # planner LLM and the multimodal call are deterministic regardless of
    # whether a previous test already imported api.main.
    from api.main import app
    import agent.graph as graph_module
    import modules.vision.engine as vision_engine_module

    fake = install_fake(monkeypatch)
    monkeypatch.setattr(graph_module, "generate", fake.generate)
    monkeypatch.setattr(
        vision_engine_module, "generate_with_vision", fake.generate_with_vision
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t"
    ) as client:
        # Beat 1: health
        r = await client.get("/api/v1/health")
        assert r.status_code == 200, r.text
        health = r.json()
        assert health["gemma4_model_configured"].startswith("gemma-4-")

        # Beat 2: sync bundle (offline setup).
        # httpx auto-decodes Content-Encoding: gzip, so stream raw bytes to
        # validate the server actually emitted gzip before decompressing.
        async with client.stream(
            "GET",
            "/api/v1/sync/bundle",
            params={"state": "Punjab", "district": "Ludhiana"},
        ) as bundle_resp:
            raw = b""
            async for chunk in bundle_resp.aiter_raw():
                raw += chunk
            assert bundle_resp.status_code == 200, raw
            assert bundle_resp.headers.get("content-encoding") == "gzip"
        body = json.loads(gzip.decompress(raw))
        assert body["bundle_version"]
        assert isinstance(body["data"]["schemes"], list) and len(body["data"]["schemes"]) > 0

        # Beat 3: online scheme query
        fake.plan_response = {
            "tools": [{"tool": "scheme", "params": {"query": "PM Fasal Bima"}}]
        }
        r = await client.post(
            "/api/v1/query",
            json={
                "farmer_id": "smoke-f1",
                "query": {
                    "text": "PM Fasal Bima kaise apply karein?",
                    "language": "hi",
                },
                "context": {
                    "connectivity": "online",
                    "device_intent": "scheme_query",
                    "location": {"district": "Ludhiana", "state": "Punjab"},
                },
            },
        )
        assert r.status_code == 200, r.text
        q = r.json()
        assert q["model_used"].startswith("gemma-4-")
        assert q["data_source"] == "live"
        assert any(t in q["tool_trace"] for t in ("scheme", "scheme_0", "safety_escalation"))

        # Beat 4: image upload + vision query
        with open("tests/fixtures/wheat_rust.jpg", "rb") as f:
            img_bytes = f.read()
        r = await client.post(
            "/api/v1/query/image",
            files={"image": ("wheat.jpg", img_bytes, "image/jpeg")},
            data={"farmer_id": "smoke-f1", "purpose": "crop_disease"},
        )
        assert r.status_code == 201, r.text
        image_ref = r.json()["image_ref"]

        fake.plan_response = {
            "tools": [{"tool": "vision", "params": {"use_image": True}}]
        }
        r = await client.post(
            "/api/v1/query",
            json={
                "farmer_id": "smoke-f1",
                "query": {
                    "text": "पत्ता पीला है",
                    "image_ref": image_ref,
                    "language": "hi",
                },
                "context": {
                    "connectivity": "online",
                    "device_intent": "crop_disease",
                    "location": {"district": "Ludhiana", "state": "Punjab"},
                },
            },
        )
        assert r.status_code == 200, r.text
        v = r.json()
        assert v["structured"]["kind"] in {"disease", "general"}
        assert v["model_used"].startswith("gemma-4-")
