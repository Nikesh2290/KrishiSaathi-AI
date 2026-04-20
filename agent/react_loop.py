"""LangGraph ReAct-style loop: route → plan → tools → synthesize."""

from __future__ import annotations

import json
import logging
import operator
from typing import Annotated, Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from agent.connectivity_router import data_source_for_route, resolve_route
from agent.dispatcher import DispatchContext, run_tools
from agent.gemma_client import generate
from agent.planner import plan_tools
from config.settings import get_settings
from models.request import AgentRequest

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    request: AgentRequest
    route: str
    data_source: str
    prefer_local: bool
    offline: bool
    tool_plan: List[Dict[str, Any]]
    tool_results: Dict[str, Any]
    tool_trace: Annotated[List[str], operator.add]
    draft_text: str


async def node_route(state: AgentState) -> Dict[str, Any]:
    settings = get_settings()
    req = state["request"]
    route = await resolve_route(req, settings)
    ds = data_source_for_route(route, req)
    offline = ds == "offline"
    prefer_local = route == "local" or offline
    return {
        "route": route,
        "data_source": ds,
        "prefer_local": prefer_local,
        "offline": offline,
    }


async def node_plan(state: AgentState) -> Dict[str, Any]:
    settings = get_settings()
    req = state["request"]
    plan = await plan_tools(req, prefer_local=bool(state.get("prefer_local")), settings=settings)
    return {"tool_plan": plan}


async def node_tools(state: AgentState) -> Dict[str, Any]:
    settings = get_settings()
    req = state["request"]
    ctx = DispatchContext(
        request=req,
        prefer_local=bool(state.get("prefer_local")),
        offline=bool(state.get("offline")),
        settings=settings,
    )
    results, trace = await run_tools(state.get("tool_plan") or [], ctx)
    return {"tool_results": results, "tool_trace": trace}


async def node_synthesize(state: AgentState) -> Dict[str, Any]:
    settings = get_settings()
    req = state["request"]
    payload = json.dumps(
        {"tools": state.get("tool_results"), "user_query": req.query.text},
        ensure_ascii=False,
    )[:12000]
    messages = [
        {
            "role": "system",
            "content": "You are KrishiSaathi. Summarize tool results for the farmer. Be practical. Match farmer language (Hindi/Hinglish if query is Hindi).",
        },
        {"role": "user", "content": payload},
    ]
    try:
        draft = await generate(messages, prefer_local=bool(state.get("prefer_local")), settings=settings)
    except Exception as e:
        logger.exception("Synthesize failed: %s", e)
        draft = "यहाँ उपलब्ध जानकारी के आधार पर सुझाव दिए गए हैं। कृपया स्थानीय कृषि अधिकारी से पुष्टि करें।"
    return {"draft_text": draft}


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("route", node_route)
    g.add_node("plan", node_plan)
    g.add_node("tools", node_tools)
    g.add_node("synthesize", node_synthesize)
    g.set_entry_point("route")
    g.add_edge("route", "plan")
    g.add_edge("plan", "tools")
    g.add_edge("tools", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


_compiled = None


def get_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled


async def run_graph(req: AgentRequest) -> AgentState:
    graph = get_graph()
    initial: AgentState = {"request": req}
    out = await graph.ainvoke(initial)
    return out  # type: ignore[return-value]


async def stream_events(req: AgentRequest):
    """Yield dict events for SSE: step + partial state."""
    yield {"step": "route", "status": "running"}
    settings = get_settings()
    route = await resolve_route(req, settings)
    ds = data_source_for_route(route, req)
    offline = ds == "offline"
    prefer_local = route == "local" or offline
    yield {"step": "route", "status": "done", "route": route, "data_source": ds}

    yield {"step": "plan", "status": "running"}
    plan = await plan_tools(req, prefer_local=prefer_local, settings=settings)
    yield {"step": "plan", "status": "done", "tools": [p.get("tool") for p in plan]}

    yield {"step": "tools", "status": "running"}
    ctx = DispatchContext(request=req, prefer_local=prefer_local, offline=offline, settings=settings)
    results, trace = await run_tools(plan, ctx)
    yield {"step": "tools", "status": "done", "tool_trace": trace}

    yield {"step": "synthesize", "status": "running"}
    payload = json.dumps({"tools": results, "user_query": req.query.text}, ensure_ascii=False)[:12000]
    messages = [
        {
            "role": "system",
            "content": "You are KrishiSaathi. Summarize tool results for the farmer.",
        },
        {"role": "user", "content": payload},
    ]
    try:
        draft = await generate(messages, prefer_local=prefer_local, settings=settings)
    except Exception as e:
        yield {"step": "synthesize", "status": "error", "error": str(e)}
        return
    yield {"step": "synthesize", "status": "done", "draft": draft[:500]}
