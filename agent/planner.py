"""Planner: produce ordered tool calls (JSON) from farmer request."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from agent.gemma_client import generate
from config.settings import Settings, get_settings
from models.request import AgentRequest

logger = logging.getLogger(__name__)

PLANNER_PROMPT = """You are the planning component for KrishiSaathi, an AI for Indian farmers.
Return ONLY a JSON object (no markdown) with this shape:
{"tools":[{"tool":"TOOL_NAME","params":{}}]}
Valid TOOL_NAME values:
- climate — params: lat (number), lng (number), crop (string)
- vision — params: use_image (boolean) — set true if user uploaded an image for disease
- scheme — params: query (string)
- crop_planner — params: season (string), optional crop (string)
- financial — params: {} (uses farmer profile)
- market — params: crop (string), district (string)

Rules:
- If device_intent is crop_disease or query mentions disease/yellow/rust/pest OR image present, include vision with use_image true if image present.
- If device_intent is weather or rain, include climate.
- If scheme/subsidy/PM-KISAN/KCC, include scheme.
- Prefer at most 3 tools. Order: vision first if needed, then climate, then others.
"""


def _heuristic_plan(req: AgentRequest) -> List[Dict[str, Any]]:
    text = (req.query.text or "").lower()
    intent = (req.context.device_intent or "general").lower()
    lat = float(req.context.location.get("lat") or 20.59)
    lng = float(req.context.location.get("lng") or 78.96)
    crop = "wheat"
    if req.query.text:
        for c in ("wheat", "rice", "cotton", "mustard", "maize", "soybean", "potato", "onion"):
            if c in text:
                crop = c
                break
    tools: List[Dict[str, Any]] = []
    if req.query.image_b64 or "disease" in intent or any(
        k in text for k in ("disease", "pest", "yellow", "rust", "फसल", "रोग")
    ):
        tools.append({"tool": "vision", "params": {"use_image": bool(req.query.image_b64)}})
    if "weather" in intent or any(k in text for k in ("rain", "weather", "मौसम", "बारिश")):
        tools.append({"tool": "climate", "params": {"lat": lat, "lng": lng, "crop": crop}})
    if "scheme" in intent or any(k in text for k in ("scheme", "subsidy", "pm-kisan", "kcc", "योजना")):
        tools.append({"tool": "scheme", "params": {"query": req.query.text or "government schemes"}})
    if "market" in intent or "price" in text or "मंडी" in req.query.text:
        dist = req.context.location.get("district") or "Ludhiana"
        tools.append({"tool": "market", "params": {"crop": crop, "district": str(dist)}})
    if "crop" in intent or "plan" in text or "फसल" in req.query.text:
        tools.append({"tool": "crop_planner", "params": {"season": "rabi", "crop": crop}})
    if "financial" in intent or "loan" in text or "बीमा" in req.query.text:
        tools.append({"tool": "financial", "params": {}})
    if not tools:
        tools.append({"tool": "scheme", "params": {"query": req.query.text or "PM-KISAN eligibility"}})
    return tools[:3]


async def plan_tools(req: AgentRequest, prefer_local: bool, settings: Settings | None = None) -> List[Dict[str, Any]]:
    settings = settings or get_settings()
    user = json.dumps(
        {
            "farmer_id": req.farmer_id,
            "query": req.query.model_dump(),
            "context": req.context.model_dump(),
        },
        ensure_ascii=False,
    )
    messages = [
        {"role": "system", "content": PLANNER_PROMPT},
        {"role": "user", "content": user},
    ]
    try:
        raw = await generate(messages, prefer_local=prefer_local, settings=settings)
        raw = raw.strip()
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
