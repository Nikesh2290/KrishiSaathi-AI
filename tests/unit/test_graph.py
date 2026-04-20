import pytest

from models.request import AgentRequest, ContextPayload, QueryPayload
from tests.fakes.fake_gemma_client import install_fake


@pytest.mark.asyncio
async def test_graph_online_path_calls_planner_and_tools(monkeypatch):
    install_fake(monkeypatch)
    from agent.graph import run_graph

    req = AgentRequest(
        farmer_id="f1",
        query=QueryPayload(text="PM-KISAN eligibility?", language="en"),
        context=ContextPayload(
            location={"district": "Ludhiana", "state": "Punjab"},
            connectivity="online",
            device_intent="scheme_query",
        ),
    )
    state = await run_graph(req)
    assert state["data_source"] == "live"
    assert "scheme" in state["tool_trace"]
    assert state["draft_text"]
    assert state["model_used"].startswith("gemma-4-")


@pytest.mark.asyncio
async def test_graph_offline_path_marks_data_source(monkeypatch):
    install_fake(monkeypatch)
    from agent.graph import run_graph

    req = AgentRequest(
        farmer_id="f1",
        query=QueryPayload(text="मौसम?", language="hi"),
        context=ContextPayload(
            location={"district": "Ludhiana", "state": "Punjab"},
            connectivity="offline",
            device_intent="weather",
        ),
    )
    state = await run_graph(req)
    assert state["data_source"] == "offline"


@pytest.mark.asyncio
async def test_graph_respects_max_iterations(monkeypatch):
    install_fake(monkeypatch)
    from agent.graph import run_graph

    req = AgentRequest(
        farmer_id="f1",
        query=QueryPayload(text="anything", language="en"),
        context=ContextPayload(device_intent="general"),
    )
    state = await run_graph(req)
    assert len(state["tool_trace"]) <= 6
