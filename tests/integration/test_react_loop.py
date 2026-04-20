"""React loop smoke test with mocks."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from agent.react_loop import run_graph
from models.request import AgentRequest, ContextPayload, QueryPayload


@pytest.mark.asyncio
async def test_run_graph_mock(monkeypatch):
    monkeypatch.setattr(
        "agent.react_loop.plan_tools",
        AsyncMock(return_value=[{"tool": "scheme", "params": {"query": "PM-KISAN"}}]),
    )
    monkeypatch.setattr(
        "agent.react_loop.run_tools",
        AsyncMock(return_value=({"scheme_0": {"answer": "x"}}, ["scheme"])),
    )
    monkeypatch.setattr(
        "agent.react_loop.generate",
        AsyncMock(return_value="Summary for farmer."),
    )
    req = AgentRequest(
        farmer_id="f1",
        query=QueryPayload(text="schemes", language="hi"),
        context=ContextPayload(connectivity="online"),
    )
    out = await run_graph(req)
    assert out.get("draft_text")
    assert "scheme" in (out.get("tool_trace") or [])
