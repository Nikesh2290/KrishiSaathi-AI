"""Single consolidated LangGraph StateGraph for KrishiSaathi.

Replaces agent/orchestrator.py, agent/planner.py, agent/dispatcher.py, agent/react_loop.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
import operator
import re
from dataclasses import dataclass
from collections.abc import AsyncIterator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from agent.connectivity_router import data_source_for_route, resolve_route
from agent.gemma_client import generate, generate_stream
from config.settings import Settings, get_settings
from db.persistence import persist_log_query, resolve_farmer_twin
from models.errors import KrishiHTTPException
from models.farmer import FarmerTwin
from models.request import AgentRequest
from modules.climate import engine as climate_engine
from modules.climate import offline_fallback as climate_offline
from modules.crop_planner import planner as crop_planner
from modules.financial import advisor as financial_advisor
from modules.market import engine as market_engine
from modules.scheme import navigator as scheme_navigator
from modules.vision import engine as vision_engine
from response.generator import build
from safety.layer import check as safety_check
from safety.layer import should_escalate

logger = logging.getLogger(__name__)

# ------------------------------ state ------------------------------


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
    safety_flags: List[str]
    confidence_score: float
    model_used: str
    fallback_hint: Optional[str]


# --------------------------- planner bits ---------------------------

_PLANNER_PROMPT = """You are the planning component for KrishiSaathi, an AI for Indian farmers.
Return ONLY a JSON object (no markdown) with this shape:
{"tools":[{"tool":"TOOL_NAME","params":{}}]}
Valid TOOL_NAME values:
- climate — params: lat (number), lng (number), crop (string)
- vision — params: use_image (boolean)
- scheme — params: query (string)
- crop_planner — params: season (string), optional crop (string)
- financial — params: {}
- market — params: crop (string), district (string)

Rules:
- If device_intent is crop_disease or image_ref present OR text mentions disease/pest/yellow/rust, include vision.
- If device_intent is weather or text mentions rain/weather, include climate.
- If device_intent is scheme_query or text mentions scheme/subsidy/PM-KISAN/KCC, include scheme.
- Prefer at most 3 tools.
"""


def _heuristic_plan(req: AgentRequest) -> List[Dict[str, Any]]:
    text = (req.query.text or "").lower()
    intent = (req.context.device_intent or "general").lower()
    lat = float(req.context.location.get("lat") or 20.59)
    lng = float(req.context.location.get("lng") or 78.96)
    crop = "wheat"
    for c in ("wheat", "rice", "cotton", "mustard", "maize", "soybean", "potato", "onion"):
        if c in text:
            crop = c
            break
    tools: List[Dict[str, Any]] = []
    has_image = bool(req.query.image_ref)
    if has_image or "disease" in intent or any(
        k in text for k in ("disease", "pest", "yellow", "rust", "rog", "रोग")
    ):
        tools.append({"tool": "vision", "params": {"use_image": has_image}})
    if "weather" in intent or any(k in text for k in ("rain", "weather", "मौसम", "बारिश")):
        tools.append({"tool": "climate", "params": {"lat": lat, "lng": lng, "crop": crop}})
    if "scheme" in intent or any(
        k in text for k in ("scheme", "subsidy", "pm-kisan", "kcc", "योजना")
    ):
        tools.append({"tool": "scheme", "params": {"query": req.query.text or "schemes"}})
    if "market" in intent or "price" in text or "मंडी" in req.query.text:
        dist = req.context.location.get("district") or "Ludhiana"
        tools.append({"tool": "market", "params": {"crop": crop, "district": str(dist)}})
    if "crop" in intent or "plan" in text or "फसल" in req.query.text:
        tools.append({"tool": "crop_planner", "params": {"season": "rabi", "crop": crop}})
    if "financial" in intent or "loan" in text or "बीमा" in req.query.text:
        tools.append({"tool": "financial", "params": {}})
    if not tools:
        tools.append(
            {"tool": "scheme", "params": {"query": req.query.text or "PM-KISAN eligibility"}}
        )
    return tools[:3]


async def _plan_with_llm(
    req: AgentRequest, prefer_local: bool, settings: Settings
) -> List[Dict[str, Any]]:
    user = json.dumps(
        {
            "farmer_id": req.farmer_id,
            "query": req.query.model_dump(),
            "context": req.context.model_dump(),
        },
        ensure_ascii=False,
    )
    messages = [
        {"role": "system", "content": _PLANNER_PROMPT},
        {"role": "user", "content": user},
    ]
    try:
        raw = (await generate(messages, prefer_local=prefer_local, settings=settings)).strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n", "", raw)
            raw = re.sub(r"\n```$", "", raw)
        data = json.loads(raw)
        tools = data.get("tools") or []
        if isinstance(tools, list) and tools:
            return tools[: settings.max_react_iterations]
    except Exception as e:
        logger.warning("Planner LLM failed, using heuristic: %s", e)
    return _heuristic_plan(req)


# --------------------------- dispatcher bits ---------------------------


@dataclass
class _DispatchContext:
    request: AgentRequest
    prefer_local: bool
    offline: bool
    settings: Settings


async def _with_timeout(coro, seconds: float):
    return await asyncio.wait_for(coro, timeout=seconds)


def _dispatch_timeout(settings: Settings, name: str, ctx: _DispatchContext) -> float:
    if name == "vision":
        return float(settings.vision_timeout_seconds)
    if name == "climate":
        return (
            float(settings.io_tool_timeout_seconds)
            if ctx.offline
            else float(settings.climate_timeout_seconds)
        )
    if name in ("scheme", "crop_planner", "financial"):
        return float(settings.llm_tool_timeout_seconds)
    if name == "market":
        return float(settings.io_tool_timeout_seconds)
    return float(settings.tool_timeout_seconds)


async def _dispatch_one(
    name: str, params: Dict[str, Any], ctx: _DispatchContext
) -> Dict[str, Any]:
    settings = ctx.settings
    req = ctx.request
    twin = await resolve_farmer_twin(req.farmer_id, req.context.connectivity, settings)
    timeout = _dispatch_timeout(settings, name, ctx)

    try:
        if name == "climate":
            lat = float(params.get("lat") or req.context.location.get("lat") or 30.65)
            lng = float(params.get("lng") or req.context.location.get("lng") or 75.95)
            crop = params.get("crop") or "wheat"
            if ctx.offline:
                dist = twin.location.district if twin else (
                    req.context.location.get("district") or "Ludhiana"
                )
                return await _with_timeout(
                    asyncio.to_thread(climate_offline.offline_weather, str(dist), crop),
                    timeout,
                )
            return await _with_timeout(climate_engine.get_weather(lat, lng, crop), timeout)

        if name == "vision":
            image_ref = req.query.image_ref
            return await _with_timeout(
                vision_engine.detect_disease_by_ref(
                    image_ref,
                    ctx.prefer_local,
                    settings,
                    user_query=req.query.text or "",
                ),
                timeout,
            )

        if name == "scheme":
            q = params.get("query") or req.query.text or "PM-KISAN"
            return await _with_timeout(
                scheme_navigator.find_schemes(twin, q, ctx.prefer_local, ctx.offline, settings),
                timeout,
            )

        if name == "crop_planner":
            soil = twin.land.soil_type if twin else "loamy"
            state = twin.location.state if twin else (
                req.context.location.get("state") or "Punjab"
            )
            season = params.get("season") or "rabi"
            water = twin.land.irrigation if twin else "tube_well"
            return await _with_timeout(
                crop_planner.recommend_async(
                    soil, state, season, water, ctx.prefer_local, settings
                ),
                timeout,
            )

        if name == "financial":
            t = twin or FarmerTwin(farmer_id=req.farmer_id)
            return await _with_timeout(
                financial_advisor.advise(t, ctx.prefer_local, settings=settings),
                timeout,
            )

        if name == "market":
            crop = params.get("crop") or "wheat"
            dist = params.get("district") or (
                twin.location.district if twin else
                (req.context.location.get("district") or "Ludhiana")
            )
            return await _with_timeout(
                asyncio.to_thread(market_engine.get_prices, crop, str(dist)),
                timeout,
            )

    except asyncio.TimeoutError:
        logger.warning("Tool %s timed out", name)
        return {"error": "timeout", "tool": name}
    except KrishiHTTPException:
        # Typed HTTP errors (e.g. IMAGE_REF_EXPIRED) must surface to the
        # FastAPI exception handler with their proper error envelope.
        raise
    except Exception as e:
        logger.exception("Tool %s failed: %s", name, e)
        return {"error": str(e), "tool": name}

    return {"error": "unknown_tool", "tool": name}


async def _run_tools(
    tools: List[Dict[str, Any]], ctx: _DispatchContext
) -> tuple[Dict[str, Any], List[str]]:
    results: Dict[str, Any] = {}
    trace: List[str] = []
    for i, step in enumerate(tools):
        name = step.get("tool") or step.get("name")
        if not name:
            continue
        params = step.get("params") or {}
        trace.append(name)
        results[f"{name}_{i}"] = await _dispatch_one(name, params, ctx)
    return results, trace


# ------------------------------ nodes ------------------------------


async def node_route(state: AgentState) -> Dict[str, Any]:
    settings = get_settings()
    req = state["request"]
    route = await resolve_route(req, settings)
    ds = data_source_for_route(route, req)
    offline = ds == "offline"
    return {
        "route": route,
        "data_source": ds,
        "prefer_local": offline,
        "offline": offline,
    }


async def node_plan(state: AgentState) -> Dict[str, Any]:
    settings = get_settings()
    req = state["request"]
    plan = await _plan_with_llm(req, prefer_local=bool(state.get("prefer_local")), settings=settings)
    return {"tool_plan": plan}


async def node_tools(state: AgentState) -> Dict[str, Any]:
    settings = get_settings()
    req = state["request"]
    ctx = _DispatchContext(
        request=req,
        prefer_local=bool(state.get("prefer_local")),
        offline=bool(state.get("offline")),
        settings=settings,
    )
    results, trace = await _run_tools(state.get("tool_plan") or [], ctx)
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
            "content": (
                "You are KrishiSaathi. Summarize tool results for the farmer. "
                "Be practical. Match farmer language (Hindi/Hinglish if query is Hindi).\n"
                "If a vision tool returned is_agricultural=false: briefly describe what the image "
                "shows using the description field, then politely explain your specialization "
                "(crop disease, soil, schemes, weather, farm advice) using specialization_note—"
                "do not pretend it is a crop disease."
            ),
        },
        {"role": "user", "content": payload},
    ]
    model_used = (
        settings.ollama_model
        if state.get("prefer_local")
        else settings.ai_studio_model
    )
    try:
        draft = await generate(
            messages, prefer_local=bool(state.get("prefer_local")), settings=settings
        )
    except Exception as e:
        logger.exception("Synthesize failed: %s", e)
        draft = (
            "यहाँ उपलब्ध जानकारी के आधार पर सुझाव दिए गए हैं। "
            "कृपया स्थानीय कृषि अधिकारी से पुष्टि करें।"
        )
    return {"draft_text": draft, "model_used": model_used}


def _vision_confidence(tool_results: Dict[str, Any]) -> Optional[float]:
    for v in tool_results.values():
        if isinstance(v, dict) and "confidence" in v:
            try:
                return float(v["confidence"])
            except (TypeError, ValueError):
                return None
    return None


async def node_safety(state: AgentState) -> Dict[str, Any]:
    settings = get_settings()
    tool_results = state.get("tool_results") or {}
    draft = state.get("draft_text") or ""
    vc = _vision_confidence(tool_results)
    sr = safety_check(tool_results, draft, vision_confidence=vc)

    score = vc if vc is not None else 0.75
    model_used = state.get("model_used") or ""
    trace = list(state.get("tool_trace") or [])

    if (
        not state.get("offline")
        and should_escalate(score, settings.confidence_threshold_low)
        and not state.get("_escalated")  # type: ignore[typeddict-item]
    ):
        payload = json.dumps(
            {"tools": tool_results, "user_query": state["request"].query.text},
            ensure_ascii=False,
        )[:12000]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are KrishiSaathi (heavy reasoning pass). Produce a precise, "
                    "cited, farmer-friendly answer."
                ),
            },
            {"role": "user", "content": payload},
        ]
        try:
            draft = await generate(messages, prefer_local=False, settings=settings, heavy=True)
            model_used = settings.ai_studio_model_heavy
            trace.append("safety_escalation")
            score = max(score, 0.82)
        except Exception as e:
            logger.warning("Escalation to heavy model failed: %s", e)

    return {
        "draft_text": sr.modified_text if sr.modified_text else draft,
        "safety_flags": sr.flags,
        "confidence_score": round(score, 3),
        "model_used": model_used,
        "tool_trace": trace,
    }


async def node_fallback_hint(state: AgentState) -> Dict[str, Any]:
    hint: Optional[str] = None
    if state.get("offline") and (state.get("confidence_score") or 0.0) < 0.7:
        hint = "USE_ONDEVICE"
    return {"fallback_hint": hint}


# ------------------------------ graph ------------------------------


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("route", node_route)
    g.add_node("plan", node_plan)
    g.add_node("tools", node_tools)
    g.add_node("synthesize", node_synthesize)
    g.add_node("safety", node_safety)
    g.add_node("respond", node_fallback_hint)
    g.set_entry_point("route")
    g.add_edge("route", "plan")
    g.add_edge("plan", "tools")
    g.add_edge("tools", "synthesize")
    g.add_edge("synthesize", "safety")
    g.add_edge("safety", "respond")
    g.add_edge("respond", END)
    return g.compile()


_compiled = None


def get_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled


async def run_graph(req: AgentRequest) -> AgentState:
    graph = get_graph()
    return await graph.ainvoke({"request": req})  # type: ignore[return-value]


async def run_graph_stream(req: AgentRequest) -> AsyncIterator[tuple[str, str]]:
    """Yield SSE-friendly (event_name, payload_json_line) tuples.

    Pipeline runs route/plan/tools, then streams synthesis tokens, then applies
    the same safety + fallback_hint logic as ``run_graph``. The ``done``
    payload is ``AgentResponse`` JSON; clients should treat ``text`` as
    authoritative (may differ from streamed tokens after safety escalation).
    """
    settings = get_settings()
    yield ("status", json.dumps({"stage": "routing"}, ensure_ascii=False))

    state: AgentState = {"request": req}  # type: ignore[assignment]
    state.update(await node_route(state))
    yield ("status", json.dumps({"stage": "planning"}, ensure_ascii=False))
    state.update(await node_plan(state))
    yield ("status", json.dumps({"stage": "tools"}, ensure_ascii=False))
    state.update(await node_tools(state))

    yield ("status", json.dumps({"stage": "synthesizing"}, ensure_ascii=False))
    rq = req
    payload = json.dumps(
        {"tools": state.get("tool_results"), "user_query": rq.query.text},
        ensure_ascii=False,
    )[:12000]
    messages = [
        {
            "role": "system",
            "content": (
                "You are KrishiSaathi. Summarize tool results for the farmer. "
                "Be practical. Match farmer language (Hindi/Hinglish if query is Hindi).\n"
                "If a vision tool returned is_agricultural=false: briefly describe what the image "
                "shows using the description field, then politely explain your specialization "
                "(crop disease, soil, schemes, weather, farm advice) using specialization_note—"
                "do not pretend it is a crop disease."
            ),
        },
        {"role": "user", "content": payload},
    ]
    prefer_local = bool(state.get("prefer_local"))
    model_used = settings.ollama_model if prefer_local else settings.ai_studio_model

    draft_acc = ""
    try:
        async for chunk in generate_stream(
            messages, prefer_local=prefer_local, settings=settings
        ):
            draft_acc += chunk
            yield ("token", json.dumps({"text": chunk}, ensure_ascii=False))
    except Exception as e:
        logger.exception("Streaming synthesize failed: %s", e)
        fb = (
            "यहाँ उपलब्ध जानकारी के आधार पर सुझाव दिए गए हैं। "
            "कृपया स्थानीय कृषि अधिकारी से पुष्टि करें।"
        )
        draft_acc = fb
        yield ("token", json.dumps({"text": fb}, ensure_ascii=False))

    state["draft_text"] = draft_acc
    state["model_used"] = model_used

    state.update(await node_safety(state))
    state.update(await node_fallback_hint(state))

    resp = build(
        draft_text=state.get("draft_text") or "",
        tool_results=state.get("tool_results") or {},
        tool_trace=list(state.get("tool_trace") or []),
        data_source=str(state.get("data_source") or "live"),
        language=rq.query.language,
        safety_flags=list(state.get("safety_flags") or []),
        model_used=str(state.get("model_used") or settings.ai_studio_model),
        confidence_score=float(state.get("confidence_score") or 0.5),
        fallback_hint=state.get("fallback_hint"),  # type: ignore[arg-type]
    )
    try:
        await persist_log_query(
            rq.farmer_id,
            rq.query.text,
            resp.structured.kind,
            resp.text[:2000],
            resp.data_source,
            rq.context.connectivity,
            settings,
        )
    except Exception as e:
        logger.warning("log_query failed: %s", e)

    yield ("done", resp.model_dump_json())
