"""Map planner tool names to module calls with timeout."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Dict, List

from config.settings import Settings, get_settings
from db.sqlite_client import get_farmer_twin
from models.farmer import FarmerTwin
from models.request import AgentRequest
from modules.climate import engine as climate_engine
from modules.climate import offline_fallback as climate_offline
from modules.crop_planner import planner as crop_planner
from modules.financial import advisor as financial_advisor
from modules.market import engine as market_engine
from modules.scheme import navigator as scheme_navigator
from modules.vision import engine as vision_engine

logger = logging.getLogger(__name__)


@dataclass
class DispatchContext:
    request: AgentRequest
    prefer_local: bool
    offline: bool
    settings: Settings


async def _with_timeout(coro: Awaitable[Any], seconds: float) -> Any:
    return await asyncio.wait_for(coro, timeout=seconds)


async def dispatch_tool(
    name: str,
    params: Dict[str, Any],
    ctx: DispatchContext,
) -> Dict[str, Any]:
    settings = ctx.settings
    req = ctx.request
    twin = await get_farmer_twin(req.farmer_id, settings)
    timeout = settings.tool_timeout_seconds

    try:
        if name == "climate":
            lat = float(params.get("lat") or req.context.location.get("lat") or 30.65)
            lng = float(params.get("lng") or req.context.location.get("lng") or 75.95)
            crop = params.get("crop") or "wheat"
            if ctx.offline:
                dist = twin.location.district if twin else "Ludhiana"
                return await _with_timeout(
                    asyncio.to_thread(climate_offline.offline_weather, str(dist), crop),
                    timeout,
                )
            return await _with_timeout(climate_engine.get_weather(lat, lng, crop), timeout)

        if name == "vision":
            return await _with_timeout(
                vision_engine.detect_disease(req.query.image_b64, ctx.prefer_local, settings),
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
            state = twin.location.state if twin else "Punjab"
            season = params.get("season") or "rabi"
            water = twin.land.irrigation if twin else "tube_well"
            return await _with_timeout(
                crop_planner.recommend_async(soil, state, season, water, ctx.prefer_local, settings),
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
            dist = params.get("district") or (twin.location.district if twin else "Ludhiana")
            return await asyncio.to_thread(market_engine.get_prices, crop, str(dist))

    except asyncio.TimeoutError:
        logger.warning("Tool %s timed out", name)
        return {"error": "timeout", "tool": name}
    except Exception as e:
        logger.exception("Tool %s failed: %s", name, e)
        return {"error": str(e), "tool": name}

    return {"error": "unknown_tool", "tool": name}


async def run_tools(
    tools: List[Dict[str, Any]],
    ctx: DispatchContext,
) -> tuple[Dict[str, Any], List[str]]:
    results: Dict[str, Any] = {}
    trace: List[str] = []
    for i, step in enumerate(tools):
        name = step.get("tool") or step.get("name")
        if not name:
            continue
        params = step.get("params") or {}
        key = f"{name}_{i}"
        trace.append(name)
        results[key] = await dispatch_tool(name, params, ctx)
    return results, trace
