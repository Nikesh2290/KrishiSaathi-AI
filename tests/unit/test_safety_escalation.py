import json

import pytest

from models.request import AgentRequest, ContextPayload, QueryPayload
from tests.fakes.fake_gemma_client import FakeGemmaClient, install_fake


def _patch_graph_generate(monkeypatch, fake):
    """Patch agent.graph's local `generate` binding.

    agent.graph imports `generate` via `from agent.gemma_client import generate`,
    so the symbol is rebound at module load. Patching `agent.gemma_client.generate`
    alone does not affect calls inside agent.graph once it has been imported
    (which happens in earlier tests during a full suite run).
    """
    import agent.graph as graph_mod

    monkeypatch.setattr(graph_mod, "generate", fake.generate, raising=False)


@pytest.mark.asyncio
async def test_low_confidence_escalates_to_heavy_model(monkeypatch):
    fake = FakeGemmaClient()
    fake.plan_response = {"tools": [{"tool": "vision", "params": {"use_image": True}}]}
    install_fake(monkeypatch, fake)

    # Force vision tool to return low-confidence result by stubbing the vision engine.
    async def fake_vision(image_ref, prefer_local, settings, user_query="", **kwargs):
        return {"disease": "Unknown", "confidence": 0.55, "treatment": []}

    import modules.vision.engine as ve
    monkeypatch.setattr(ve, "detect_disease_by_ref", fake_vision)

    from agent.graph import run_graph
    _patch_graph_generate(monkeypatch, fake)

    req = AgentRequest(
        farmer_id="f1",
        query=QueryPayload(text="yellow leaves", image_ref="img_fake"),
        context=ContextPayload(
            connectivity="online",
            device_intent="crop_disease",
            location={"district": "Ludhiana", "state": "Punjab"},
        ),
    )
    state = await run_graph(req)

    assert state["model_used"] == "gemma-4-31b-it"
    assert "safety_escalation" in state["tool_trace"]
    assert any(c.get("heavy") for c in fake.calls)


@pytest.mark.asyncio
async def test_high_confidence_does_not_escalate(monkeypatch):
    fake = FakeGemmaClient()
    fake.plan_response = {"tools": [{"tool": "vision", "params": {"use_image": True}}]}
    install_fake(monkeypatch, fake)

    async def fake_vision(image_ref, prefer_local, settings, user_query="", **kwargs):
        return {"disease": "Yellow Rust", "confidence": 0.92, "treatment": ["X"]}

    import modules.vision.engine as ve
    monkeypatch.setattr(ve, "detect_disease_by_ref", fake_vision)

    from agent.graph import run_graph
    _patch_graph_generate(monkeypatch, fake)

    req = AgentRequest(
        farmer_id="f1",
        query=QueryPayload(text="yellow leaves", image_ref="img_fake"),
        context=ContextPayload(
            connectivity="online",
            device_intent="crop_disease",
            location={"district": "Ludhiana", "state": "Punjab"},
        ),
    )
    state = await run_graph(req)

    assert state["model_used"] == "gemma-4-26b-a4b-it"
    assert "safety_escalation" not in state["tool_trace"]
    assert not any(c.get("heavy") for c in fake.calls)
