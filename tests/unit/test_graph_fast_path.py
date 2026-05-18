"""Fast routing helpers (Tier 2 direct LLM gate + voice planner prompt)."""

from __future__ import annotations

import json

import pytest

from agent import graph as graph_module
from agent.graph import _PLANNER_PROMPT_VOICE, _plan_with_llm
from models.request import AgentRequest, ContextPayload, QueryPayload


def test_is_nontool_true_without_tool_keywords():
    req = AgentRequest(
        farmer_id="f1",
        query=QueryPayload(text="capital of France?", language="en"),
        context=ContextPayload(),
    )
    assert graph_module._is_nontool_query(req, []) is True


def test_is_nontool_false_when_weather_keyword():
    req = AgentRequest(
        farmer_id="f1",
        query=QueryPayload(text="kal mausam kaisa rahega", language="hi"),
        context=ContextPayload(),
    )
    assert graph_module._is_nontool_query(req, []) is False


def test_is_nontool_false_with_image_ref():
    req = AgentRequest(
        farmer_id="f1",
        query=QueryPayload(text="kaisa hai", language="hi", image_ref="img_abc"),
        context=ContextPayload(),
    )
    assert graph_module._is_nontool_query(req, []) is False


@pytest.mark.asyncio
async def test_plan_voice_uses_short_prompt(monkeypatch):
    calls: list[list[dict]] = []

    async def fake_generate(messages, prefer_local=False, settings=None, **kwargs):
        calls.append(messages)
        return json.dumps(
            {"tools": [{"tool": "climate", "params": {"lat": 30.0, "lng": 75.0, "crop": "wheat"}}]}
        )

    monkeypatch.setattr(graph_module, "generate", fake_generate)

    async def fake_twin(*a, **k):
        return None

    monkeypatch.setattr(graph_module, "resolve_farmer_twin", fake_twin)

    req = AgentRequest(
        farmer_id="f1",
        query=QueryPayload(text="weather kal", language="hi"),
        context=ContextPayload(device_intent="voice"),
    )
    plan = await _plan_with_llm(req, prefer_local=False, settings=graph_module.get_settings(), chat_history=[])

    assert calls and calls[0][0]["role"] == "system"
    assert calls[0][0]["content"] == _PLANNER_PROMPT_VOICE
    assert plan and plan[0]["tool"] == "climate"
