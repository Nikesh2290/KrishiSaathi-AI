"""SSE progress frames from run_graph_stream."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from agent.graph import run_graph_stream
from models.request import AgentRequest, ContextPayload, QueryPayload


async def _collect_stream(req: AgentRequest) -> list[tuple[str, dict | None]]:
    out: list[tuple[str, dict | None]] = []
    async for part in run_graph_stream(req):
        out.append(part)
    return out


@pytest.mark.asyncio
async def test_stream_emits_routing_thinking_and_direct_llm(monkeypatch):
    async def fake_gs(_messages, prefer_local=False, settings=None):
        yield "ok"

    monkeypatch.setattr("agent.graph.generate_stream", fake_gs)
    monkeypatch.setattr("agent.graph.resolve_farmer_twin", AsyncMock(return_value=None))
    monkeypatch.setattr("agent.graph.persist_log_query", AsyncMock())
    monkeypatch.setattr("agent.graph._load_chat_history", AsyncMock(return_value=[]))

    req = AgentRequest(
        farmer_id="f1",
        conversation_id="conv-1",
        query=QueryPayload(text="what is wheat rust", language="en"),
        context=ContextPayload(device_intent="voice"),
    )
    parts = await _collect_stream(req)
    types = [p[0] for p in parts]
    tools = [
        p[1]["data"]["tool"]
        for p in parts
        if p[0] == "data-tool" and p[1] and p[1].get("data", {}).get("status") == "started"
    ]
    assert "routing" in tools
    assert "thinking" in tools
    assert "direct_llm" in tools
    assert "data-stage" in types


@pytest.mark.asyncio
async def test_stream_tool_path_emits_named_tools(monkeypatch):
    async def fake_gs(_messages, prefer_local=False, settings=None):
        yield "rain likely"

    async def fake_tools(plan, ctx):
        return {"climate_0": {"summary": "dry"}}, ["climate"]

    monkeypatch.setattr("agent.graph.generate_stream", fake_gs)
    monkeypatch.setattr("agent.graph._run_tools", fake_tools)
    monkeypatch.setattr("agent.graph.resolve_farmer_twin", AsyncMock(return_value=None))
    monkeypatch.setattr("agent.graph.persist_log_query", AsyncMock())
    monkeypatch.setattr("agent.graph._load_chat_history", AsyncMock(return_value=[]))

    req = AgentRequest(
        farmer_id="f1",
        conversation_id="conv-2",
        query=QueryPayload(text="kal mausam kaisa rahega", language="hi"),
        context=ContextPayload(device_intent="voice"),
    )
    parts = await _collect_stream(req)
    tools_started = [
        p[1]["data"]["tool"]
        for p in parts
        if p[0] == "data-tool" and p[1] and p[1].get("data", {}).get("status") == "started"
    ]
    assert "climate" in tools_started
    assert "synthesizing" in tools_started
    assert "direct_llm" not in tools_started
