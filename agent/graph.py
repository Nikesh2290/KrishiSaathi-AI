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
from uuid import uuid4
from collections.abc import AsyncIterator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from models.response import AgentResponse

from langgraph.graph import END, StateGraph

from agent.connectivity_router import data_source_for_route, resolve_route
from agent.gemma_client import generate, generate_stream
from agent.history import build_chat_history_context
from config.settings import Settings, get_settings
from db.persistence import persist_log_query, resolve_farmer_twin
from db.sqlite_client import get_last_n_turns
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

CLARIFY_TOOL = "__clarify__"
SMALLTALK_TOOL = "__smalltalk__"
_ALLOWED_PLANNER_TOOLS = frozenset(
    {
        "climate",
        "vision",
        "scheme",
        "crop_planner",
        "financial",
        "market",
        "general_qa",
    }
)

_LANGUAGE_RULE = (
    "LANGUAGE RULE (MANDATORY — follow before anything else): "
    "Read the user's message text. Identify the language they used: "
    "Hindi (Devanagari script), Hinglish (Hindi written in Roman/Latin script), or English. "
    "Write your ENTIRE response in that SAME language. "
    "If they wrote in Devanagari Hindi → reply in Devanagari Hindi. "
    "If they wrote in Hinglish (Roman-script Hindi) → reply in Hinglish. "
    "If they wrote in English → reply in English. "
    "Never switch languages mid-response. Technical terms (NPK, pH, KCC) may appear as-is."
)


def _system_with_language_rule(base_prompt: str) -> str:
    return f"{_LANGUAGE_RULE}\n\n{base_prompt}"


_SYNTHESIS_SYSTEM_PROMPT = """You are KrishiSaathi, a smart and helpful assistant for Indian farmers.

You are given:
1. The user's exact query
2. Results from data tools (weather, market prices, crop recommendations, schemes, financial numbers, disease detection, etc.)

Your job:
- Answer the user's question directly and completely first.
- If any tool result is relevant and helpful to the question, weave it naturally into your answer.
- If a tool result is NOT relevant to what the user asked, ignore it entirely (do not mention unrelated tools).
- If a general_qa tool result is present with an "answer" field, treat that as the core answer — keep it as-is unless other tools clearly add useful facts for this query.
- Never lead with unrelated suggestions or alternatives before answering what was asked.

If a vision tool returned is_agricultural=false: briefly describe what the image shows using the description field, then politely explain your specialization (crop disease, soil, schemes, weather, farm advice) using specialization_note—do not pretend it is a crop disease.

CHAT HISTORY (when present in the JSON payload): previous turns; assistant replies may be truncated to 250 chars.
Use only to understand what the current question refers to. Never re-answer old questions unless the user asks again.
If details appear cut off, answer what is clear and invite a follow-up.
If chat_history is empty or absent, ignore this section entirely.

When farmer_profile is present in the JSON payload, use the farmer's name naturally when addressing them (respectfully); if name is missing or empty, use neutral address."""

_SMALLTALK_SYSTEM_PROMPT = """You are KrishiSaathi, a warm and friendly AI companion for Indian farmers.
The farmer has sent a greeting, thanks, or social message — not a farming question.

Rules:
- If farmer_profile.name is present and non-empty, use it once naturally and respectfully (e.g. "Namaste, Ramesh ji!").
- If name is missing or empty, use a neutral warm address based on language:
  Hindi/Hinglish: "Namaste!" | English: "Hello there!"
- Keep reply to 1-2 short sentences.
- End with ONE gentle open question inviting their real farming need
  (e.g. "Aaj kaise madad kar sakta hoon?" / "How can I help you today?").
- Never invent or assume farming details."""

# ------------------------------ state ------------------------------


class AgentState(TypedDict, total=False):
    request: AgentRequest
    route: str
    data_source: str
    prefer_local: bool
    offline: bool
    chat_history: List[Dict[str, Any]]
    tool_plan: List[Dict[str, Any]]
    tool_results: Dict[str, Any]
    tool_trace: Annotated[List[str], operator.add]
    draft_text: str
    safety_flags: List[str]
    confidence_score: float
    model_used: str
    fallback_hint: Optional[str]


def _farmer_profile_for_llm(twin: Optional[FarmerTwin]) -> Optional[Dict[str, Any]]:
    if not twin:
        return None
    return {
        "name": twin.name,
        "location": twin.location.model_dump(),
        "land": twin.land.model_dump(),
        "current_crops": twin.current_crops,
        "preferred_language": twin.preferred_language,
    }


# --------------------------- planner bits ---------------------------

_PLANNER_PROMPT = """You are the intent classifier for KrishiSaathi, an AI assistant for Indian farmers.
Use semantic understanding: users may write English, Hindi, Hinglish, or common typos — infer MEANING.

Return ONLY valid JSON (no markdown). Exactly ONE of these shapes:
{"tools":[{"tool":"TOOL_NAME","params":{}}]}
{"clarify": true, "question": "<one short clarifying question in the user's language>"}
{"smalltalk": true}

Allowed TOOL_NAME values and params:
- general_qa — params: {"query": "<echo user question>"}
  How-to / agronomy / planting / sowing / harvesting / irrigation / storage / pest advice WITHOUT an attached crop image.
  Examples: "how to plant potato", "aloo kaise lagayein", "plnat tomato" (typo), "when to irrigate wheat"
  NOT for choosing which crop is best for the whole season (that is crop_planner).

- crop_planner — params: {"season": "<rabi|kharif|zaid or best guess>", "crop": "<mentioned crop or wheat>"}
  User wants WHAT to grow / crop recommendation / rotation planning for the season.
  Examples: "best crop this season", "kya ugaayein is baar", "which crop should I grow"
  CRITICAL: "plant" / "planting" / how-to-grow a named crop → general_qa, NOT crop_planner.

- climate — params: {"lat": number, "lng": number, "crop": string}
  Weather, rain, forecast, temperature for farming decisions.

- vision — params: {"use_image": true|false}
  Disease / pest on leaves / identify problem FROM IMAGE. If query.image_ref is present OR device_intent suggests crop disease, include vision with use_image true when image_ref exists.

- scheme — params: {"query": string}
  Government schemes, subsidies, PM-KISAN, KCC rules, eligibility.

- market — params: {"crop": string, "district": string}
  Mandi price, market rate, selling price.

- financial — params: {}
  Loans, KCC limits, crop insurance, premium estimates.

Rules:
- Prefer at most 3 tools in the tools array. Fewer is better when one tool suffices.
- If query.image_ref is present and user may be asking about the photo → include vision first with use_image true.
- Use farmer_profile (name, location, land, crops, preferred_language) from input only to disambiguate — do not invent facts.

WHEN TO USE smalltalk (before checking tools or clarify):
- Pure greeting: "hi", "hello", "namaste", "hii", "helo", etc.
- Pure thanks: "thanks", "thank you", "shukriya", "dhanyawad", etc.
- Pure acknowledgement or leave-taking: "ok", "bye", "okay", "theek hai", "acha", "got it", etc.
- Apology with no farming question: "sorry", "maafi", etc.
- No farming intent whatsoever — ONLY a social or emotional message.
Return {"smalltalk": true}. Do NOT use smalltalk if ANY farming topic is also present (e.g. "hi, what's wheat price?" → use market tool).

WHEN TO USE clarify instead of tools:
- Query is too vague or one word with no clear farming intent ("help", "kuch batao", "problem", "potato" alone).
- Query is gibberish or too short to route confidently.
- Two very different interpretations are equally likely and profile does not resolve them.
Return {"clarify": true, "question": "..."} with ONE focused question.
- Do NOT use clarify for pure greetings, thanks, or other social-only messages — use {"smalltalk": true} instead.

WHEN NOT to clarify: if intent is reasonably clear, pick tools. Do not over-ask.

CHAT HISTORY (when chat_history in input JSON is non-empty):
- Contains last ≤3 prior turns for this conversation only. Responses may be truncated to 250 chars.
- Use ONLY to resolve references in the current query ("woh wali fasal", "us mein", "aur kya?").
- Do NOT infer tool parameters from history unless the current query clearly refers to them.
- If a key detail looks cut off, prefer clarify over guessing.
- If chat_history is empty or absent, ignore this section entirely.
"""


def _heuristic_plan(req: AgentRequest) -> List[Dict[str, Any]]:
    """Emergency fallback only when the LLM planner call fails (network/parse error)."""
    text = (req.query.text or "").lower()
    intent = (req.context.device_intent or "general").lower()
    lat = float(req.context.location.get("lat") or 20.59)
    lng = float(req.context.location.get("lng") or 78.96)
    crop = "wheat"
    for c in ("wheat", "rice", "cotton", "mustard", "maize", "soybean", "potato", "onion"):
        if c in text:
            crop = c
            break
    has_image = bool(req.query.image_ref)
    if has_image:
        return [{"tool": "vision", "params": {"use_image": True}}]
    if "weather" in intent or any(k in text for k in ("rain", "weather", "मौसम", "बारिश", "barish", "baarish", "mausam")):
        return [{"tool": "climate", "params": {"lat": lat, "lng": lng, "crop": crop}}]
    if any(k in text for k in ("scheme", "subsidy", "pm-kisan", "kcc", "योजना", "yojana")):
        return [{"tool": "scheme", "params": {"query": req.query.text or "schemes"}}]
    if "market" in intent or any(k in text for k in ("price", "mandi", "rate", "bhav", "मंडी")):
        dist = req.context.location.get("district") or "Ludhiana"
        return [{"tool": "market", "params": {"crop": crop, "district": str(dist)}}]
    if "financial" in intent or any(k in text for k in ("loan", "बीमा", "insurance", "kcc")):
        return [{"tool": "financial", "params": {}}]
    if "disease" in intent or any(k in text for k in ("disease", "pest", "yellow", "rust", "rog", "रोग")):
        return [{"tool": "vision", "params": {"use_image": has_image}}]
    return [{"tool": "general_qa", "params": {"query": req.query.text or ""}}]


def _normalize_tool_plan(raw_tools: Any, settings: Settings) -> List[Dict[str, Any]]:
    if not isinstance(raw_tools, list):
        return []
    out: List[Dict[str, Any]] = []
    for step in raw_tools:
        if not isinstance(step, dict):
            continue
        name = step.get("tool") or step.get("name")
        if not isinstance(name, str) or name not in _ALLOWED_PLANNER_TOOLS:
            continue
        params = step.get("params")
        if not isinstance(params, dict):
            params = {}
        out.append({"tool": name, "params": params})
        if len(out) >= settings.max_react_iterations:
            break
    return out


async def _load_chat_history(req: AgentRequest, settings: Settings) -> List[Dict[str, Any]]:
    """Prior turns only (SQLite); empty when no conversation_id."""
    cid = (req.conversation_id or "").strip()
    if not cid:
        return []
    try:
        turns = await get_last_n_turns(cid, n=3, settings=settings)
        return build_chat_history_context(turns)
    except Exception as e:
        logger.warning("chat history load failed: %s", e)
        return []


async def _plan_with_llm(
    req: AgentRequest,
    prefer_local: bool,
    settings: Settings,
    *,
    chat_history: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    twin = await resolve_farmer_twin(req.farmer_id, req.context.connectivity, settings)
    twin_payload = _farmer_profile_for_llm(twin)
    hist = chat_history if chat_history is not None else []
    user = json.dumps(
        {
            "farmer_id": req.farmer_id,
            "farmer_profile": twin_payload,
            "query": req.query.model_dump(),
            "context": req.context.model_dump(),
            "chat_history": hist,
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
        if isinstance(data, dict) and data.get("clarify") is True:
            q = str(data.get("question") or "").strip() or (
                "कृपया अपना सवाल थोड़ा और स्पष्ट करें — आपको फसल, मौसम, योजना या बाज़ार में से किस बारे में जानकारी चाहिए? "
                "/ Please clarify your question — do you need help with crop, weather, schemes, or market prices?"
            )
            return [{"tool": CLARIFY_TOOL, "params": {"question": q}}]
        if isinstance(data, dict) and data.get("smalltalk") is True:
            return [{"tool": SMALLTALK_TOOL, "params": {}}]
        tools = data.get("tools") if isinstance(data, dict) else None
        normalized = _normalize_tool_plan(tools, settings)
        if normalized:
            return normalized
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
    if name in ("scheme", "crop_planner", "financial", "general_qa"):
        return float(settings.llm_tool_timeout_seconds)
    if name == "market":
        return float(settings.market_tool_timeout_seconds)
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
            water = "tube_well"
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
            st = twin.location.state if twin else (
                req.context.location.get("state") or "Punjab"
            )
            return await _with_timeout(
                market_engine.get_prices(crop, str(dist), settings, state=str(st)),
                timeout,
            )

        if name == "general_qa":
            soil = twin.land.soil_type if twin else "loamy"
            st = twin.location.state if twin else (
                str(req.context.location.get("state") or "") or "India"
            )
            crops = twin.current_crops if twin else []
            name_s = (twin.name.strip() if twin and twin.name else "") or "Farmer"
            q = params.get("query") or req.query.text or ""
            msgs = [
                {
                    "role": "system",
                    "content": _system_with_language_rule(
                        f"You are an expert agronomist for Indian farmers in {st}. "
                        f"Address the farmer respectfully by name when natural: {name_s}. "
                        f"This farmer has {soil} soil and is currently growing: "
                        f"{crops or 'not specified'}. "
                        "Give the direct answer first, then practical tips."
                    ),
                },
                {"role": "user", "content": q},
            ]
            answer = await _with_timeout(
                generate(msgs, prefer_local=ctx.prefer_local, settings=settings),
                timeout,
            )
            return {"answer": answer, "source": "llm_knowledge"}

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
    prefer_local = bool(state.get("prefer_local"))
    chat_hist = await _load_chat_history(req, settings)
    plan = await _plan_with_llm(
        req,
        prefer_local=prefer_local,
        settings=settings,
        chat_history=chat_hist,
    )
    model_used = settings.ollama_model if prefer_local else settings.ai_studio_model
    return {"tool_plan": plan, "model_used": model_used, "chat_history": chat_hist}


async def node_clarify(state: AgentState) -> Dict[str, Any]:
    plan = state.get("tool_plan") or []
    question = (
        plan[0]["params"].get("question")
        if plan and isinstance(plan[0].get("params"), dict)
        else None
    ) or "कृपया अपना सवाल स्पष्ट करें। / Please clarify your question."
    settings = get_settings()
    prefer_local = bool(state.get("prefer_local"))
    model_used = state.get("model_used") or (
        settings.ollama_model if prefer_local else settings.ai_studio_model
    )
    return {
        "draft_text": question,
        "tool_results": {},
        "tool_trace": ["clarify"],
        "safety_flags": [],
        "confidence_score": 1.0,
        "model_used": model_used,
    }


async def _build_smalltalk_llm_messages(state: AgentState) -> tuple[list[Dict[str, Any]], str]:
    settings = get_settings()
    req = state["request"]
    prefer_local = bool(state.get("prefer_local"))
    twin = await resolve_farmer_twin(req.farmer_id, req.context.connectivity, settings)
    farmer_profile = _farmer_profile_for_llm(twin)
    payload = json.dumps(
        {
            "farmer_profile": farmer_profile,
            "user_message": req.query.text,
            "chat_history": state.get("chat_history") or [],
        },
        ensure_ascii=False,
    )
    messages = [
        {
            "role": "system",
            "content": _system_with_language_rule(_SMALLTALK_SYSTEM_PROMPT),
        },
        {"role": "user", "content": payload},
    ]
    model_used = settings.ollama_model if prefer_local else settings.ai_studio_model
    return messages, model_used


async def node_smalltalk(state: AgentState) -> Dict[str, Any]:
    settings = get_settings()
    prefer_local = bool(state.get("prefer_local"))
    messages, model_used = await _build_smalltalk_llm_messages(state)
    try:
        draft = await generate(messages, prefer_local=prefer_local, settings=settings)
    except Exception:
        draft = "Namaste! Aaj main aapki kya madad kar sakta hoon?"
    return {
        "draft_text": draft,
        "tool_results": {},
        "tool_trace": ["smalltalk"],
        "safety_flags": [],
        "confidence_score": 1.0,
        "model_used": model_used,
    }


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
    twin = await resolve_farmer_twin(req.farmer_id, req.context.connectivity, settings)
    farmer_profile = _farmer_profile_for_llm(twin)
    payload = json.dumps(
        {
            "farmer_profile": farmer_profile,
            "tools": state.get("tool_results"),
            "user_query": req.query.text,
            "chat_history": state.get("chat_history") or [],
        },
        ensure_ascii=False,
    )[:12000]
    messages = [
        {
            "role": "system",
            "content": _system_with_language_rule(_SYNTHESIS_SYSTEM_PROMPT),
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
        rq_esc = state["request"]
        twin_esc = await resolve_farmer_twin(
            rq_esc.farmer_id, rq_esc.context.connectivity, settings
        )
        payload = json.dumps(
            {
                "farmer_profile": _farmer_profile_for_llm(twin_esc),
                "tools": tool_results,
                "user_query": rq_esc.query.text,
                "chat_history": state.get("chat_history") or [],
            },
            ensure_ascii=False,
        )[:12000]
        messages = [
            {
                "role": "system",
                "content": _system_with_language_rule(
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


def _route_after_plan(state: AgentState) -> str:
    plan = state.get("tool_plan") or []
    if plan and plan[0].get("tool") == CLARIFY_TOOL:
        return "clarify"
    if plan and plan[0].get("tool") == SMALLTALK_TOOL:
        return "smalltalk"
    return "tools"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("route", node_route)
    g.add_node("plan", node_plan)
    g.add_node("clarify", node_clarify)
    g.add_node("smalltalk", node_smalltalk)
    g.add_node("tools", node_tools)
    g.add_node("synthesize", node_synthesize)
    g.add_node("safety", node_safety)
    g.add_node("respond", node_fallback_hint)
    g.set_entry_point("route")
    g.add_edge("route", "plan")
    g.add_conditional_edges(
        "plan",
        _route_after_plan,
        {"clarify": "clarify", "smalltalk": "smalltalk", "tools": "tools"},
    )
    g.add_edge("clarify", "respond")
    g.add_edge("smalltalk", "safety")
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


def _agent_response_metadata(resp: AgentResponse) -> dict:
    """Payload for AI SDK ``data-metadata`` part (AgentResponse without streamed text)."""
    d = resp.model_dump(mode="json")
    d.pop("text", None)
    return {"data": d}


async def run_graph_stream(
    req: AgentRequest,
) -> AsyncIterator[tuple[str, Optional[dict[str, Any]]]]:
    """Yield AI SDK UI Data Stream parts as (part_type, extra_fields_or_None).

    The HTTP layer emits ``data: {"type":<part_type>, ...}`` and ends with ``data: [DONE]``.
    Typed text deltas use ``text-start`` / ``text-delta`` / ``text-end`` with shared ``id``.
    """
    settings = get_settings()
    message_id = str(uuid4())
    text_id = f"txt_{uuid4().hex}"

    yield ("start", {"messageId": message_id})
    yield ("data-stage", {"data": {"stage": "routing"}})

    state: AgentState = {"request": req}  # type: ignore[assignment]
    state.update(await node_route(state))
    yield ("data-stage", {"data": {"stage": "planning"}})
    state.update(await node_plan(state))

    plan_early = state.get("tool_plan") or []
    if plan_early and plan_early[0].get("tool") == CLARIFY_TOOL:
        yield ("data-stage", {"data": {"stage": "clarify"}})
        state.update(await node_clarify(state))
        clarify_text = state.get("draft_text") or ""

        yield ("start-step", {})
        yield ("text-start", {"id": text_id})
        if clarify_text:
            yield ("text-delta", {"id": text_id, "delta": clarify_text})
        yield ("text-end", {"id": text_id})
        yield ("finish-step", {})

        state.update(await node_fallback_hint(state))

        rq = req
        resp = build(
            draft_text=state.get("draft_text") or "",
            tool_results={},
            tool_trace=list(state.get("tool_trace") or []),
            data_source=str(state.get("data_source") or "live"),
            language=rq.query.language,
            safety_flags=list(state.get("safety_flags") or []),
            model_used=str(state.get("model_used") or settings.ai_studio_model),
            confidence_score=float(state.get("confidence_score") or 1.0),
            fallback_hint=state.get("fallback_hint"),  # type: ignore[arg-type]
        )
        resp.conversation_id = rq.conversation_id
        try:
            await persist_log_query(
                rq.query.text,
                resp.structured.kind,
                resp.text[:2000],
                resp.data_source,
                rq.context.connectivity,
                farmer_id=rq.farmer_id,
                conversation_id=rq.conversation_id,
                settings=settings,
            )
        except Exception as e:
            logger.warning("log_query failed: %s", e)

        yield ("data-metadata", _agent_response_metadata(resp))
        yield ("finish", {})
        yield ("__done__", None)
        return

    if plan_early and plan_early[0].get("tool") == SMALLTALK_TOOL:
        yield ("data-stage", {"data": {"stage": "smalltalk"}})
        messages_st, model_used_st = await _build_smalltalk_llm_messages(state)
        prefer_local_st = bool(state.get("prefer_local"))

        yield ("start-step", {})
        yield ("text-start", {"id": text_id})

        draft_acc_st = ""
        smalltalk_fb = (
            "Namaste! Aaj main aapki kya madad kar sakta hoon?"
        )
        try:
            async for chunk in generate_stream(
                messages_st,
                prefer_local=prefer_local_st,
                settings=settings,
            ):
                draft_acc_st += chunk
                yield ("text-delta", {"id": text_id, "delta": chunk})
        except Exception as e:
            logger.exception("Streaming smalltalk failed: %s", e)
            draft_acc_st = smalltalk_fb
            yield ("text-delta", {"id": text_id, "delta": smalltalk_fb})

        yield ("text-end", {"id": text_id})
        yield ("finish-step", {})

        state["draft_text"] = draft_acc_st
        state["model_used"] = model_used_st
        state["tool_results"] = {}
        state["tool_trace"] = ["smalltalk"]
        state["safety_flags"] = []

        state.update(await node_safety(state))
        state.update(await node_fallback_hint(state))

        rq_st = req
        resp_st = build(
            draft_text=state.get("draft_text") or "",
            tool_results={},
            tool_trace=list(state.get("tool_trace") or []),
            data_source=str(state.get("data_source") or "live"),
            language=rq_st.query.language,
            safety_flags=list(state.get("safety_flags") or []),
            model_used=str(state.get("model_used") or settings.ai_studio_model),
            confidence_score=float(state.get("confidence_score") or 1.0),
            fallback_hint=state.get("fallback_hint"),  # type: ignore[arg-type]
        )
        resp_st.conversation_id = rq_st.conversation_id
        try:
            await persist_log_query(
                rq_st.query.text,
                resp_st.structured.kind,
                resp_st.text[:2000],
                resp_st.data_source,
                rq_st.context.connectivity,
                farmer_id=rq_st.farmer_id,
                conversation_id=rq_st.conversation_id,
                settings=settings,
            )
        except Exception as e:
            logger.warning("log_query failed: %s", e)

        yield ("data-metadata", _agent_response_metadata(resp_st))
        yield ("finish", {})
        yield ("__done__", None)
        return

    yield ("data-stage", {"data": {"stage": "tools"}})
    state.update(await node_tools(state))

    yield ("data-stage", {"data": {"stage": "synthesizing"}})
    rq = req
    twin_syn = await resolve_farmer_twin(rq.farmer_id, rq.context.connectivity, settings)
    payload = json.dumps(
        {
            "farmer_profile": _farmer_profile_for_llm(twin_syn),
            "tools": state.get("tool_results"),
            "user_query": rq.query.text,
            "chat_history": state.get("chat_history") or [],
        },
        ensure_ascii=False,
    )[:12000]
    messages = [
        {
            "role": "system",
            "content": _system_with_language_rule(_SYNTHESIS_SYSTEM_PROMPT),
        },
        {"role": "user", "content": payload},
    ]
    prefer_local = bool(state.get("prefer_local"))
    model_used = settings.ollama_model if prefer_local else settings.ai_studio_model

    yield ("start-step", {})
    yield ("text-start", {"id": text_id})

    draft_acc = ""
    try:
        async for chunk in generate_stream(
            messages, prefer_local=prefer_local, settings=settings
        ):
            draft_acc += chunk
            yield ("text-delta", {"id": text_id, "delta": chunk})
    except Exception as e:
        logger.exception("Streaming synthesize failed: %s", e)
        fb = (
            "यहाँ उपलब्ध जानकारी के आधार पर सुझाव दिए गए हैं। "
            "कृपया स्थानीय कृषि अधिकारी से पुष्टि करें।"
        )
        draft_acc = fb
        yield ("text-delta", {"id": text_id, "delta": fb})

    yield ("text-end", {"id": text_id})
    yield ("finish-step", {})

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
    resp.conversation_id = rq.conversation_id
    try:
        await persist_log_query(
            rq.query.text,
            resp.structured.kind,
            resp.text[:2000],
            resp.data_source,
            rq.context.connectivity,
            farmer_id=rq.farmer_id,
            conversation_id=rq.conversation_id,
            settings=settings,
        )
    except Exception as e:
        logger.warning("log_query failed: %s", e)

    yield ("data-metadata", _agent_response_metadata(resp))
    yield ("finish", {})
    yield ("__done__", None)
