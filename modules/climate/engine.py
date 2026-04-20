"""Live weather via Open-Meteo (free, no API key)."""

from __future__ import annotations

import logging
from typing import Any, Dict

import httpx

logger = logging.getLogger(__name__)

CROP_WATER: Dict[str, int] = {
    "wheat": 2,
    "rice": 3,
    "cotton": 2,
    "mustard": 2,
    "maize": 2,
    "soybean": 2,
    "potato": 2,
    "onion": 2,
}


async def get_weather(lat: float, lng: float, crop: str) -> Dict[str, Any]:
    """Fetch 7-day outlook + rain risk + irrigation hint."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lng,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
        "timezone": "Asia/Kolkata",
        "forecast_days": 7,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

    daily = data.get("daily") or {}
    probs = daily.get("precipitation_probability_max") or []
    rains = daily.get("precipitation_sum") or []
    tmax = daily.get("temperature_2m_max") or []
    tmin = daily.get("temperature_2m_min") or []

    max_rain_prob = max(probs) if probs else 0.0
    total_rain = sum(rains) if rains else 0.0

    need = CROP_WATER.get(crop.lower(), 2)
    if max_rain_prob > 60 and total_rain > 15:
        rain_risk = "high"
        irrigation_hint = "Delay irrigation; heavy rain expected."
        urgency = "medium"
    elif max_rain_prob > 35:
        rain_risk = "medium"
        irrigation_hint = (
            "Reduce irrigation slightly; watch soil moisture." if need >= 2 else "Maintain usual schedule."
        )
        urgency = "low"
    else:
        rain_risk = "low"
        irrigation_hint = (
            "Irrigate if soil is dry; low rain probability ahead." if need >= 2 else "Light irrigation may suffice."
        )
        urgency = "low"

    outlook = {
        "days": 7,
        "temp_max_c": max(tmax) if tmax else None,
        "temp_min_c": min(tmin) if tmin else None,
        "total_precip_mm_7d": round(total_rain, 1),
        "max_precip_probability_pct": round(float(max_rain_prob), 1),
    }

    return {
        "outlook": outlook,
        "rain_risk": rain_risk,
        "irrigation_hint": irrigation_hint,
        "urgency": urgency,
        "source": "open_meteo",
    }
