"""Live weather via Open-Meteo (free, no API key)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Optional

import httpx

from cache import cache_keys as ck
from cache.redis_client import get_redis_for_request, json_get_maybe, json_setex
from config.settings import Settings, get_settings

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


def wmo_weather_condition(code: Optional[int]) -> str:
    """Map WMO weather interpretation codes (Open-Meteo) to a short English label."""
    if code is None:
        return "Unknown"
    c = int(code)
    if c == 0:
        return "Clear sky"
    if c == 1:
        return "Mainly clear"
    if c == 2:
        return "Partly cloudy"
    if c == 3:
        return "Overcast"
    if c in (45, 48):
        return "Fog"
    if c in (51, 53, 55):
        return "Drizzle"
    if c in (56, 57):
        return "Freezing drizzle"
    if c in (61, 63, 65):
        return "Rain"
    if c in (66, 67):
        return "Freezing rain"
    if c in (71, 73, 75):
        return "Snow fall"
    if c == 77:
        return "Snow grains"
    if c in (80, 81, 82):
        return "Rain showers"
    if c in (85, 86):
        return "Snow showers"
    if c == 95:
        return "Thunderstorm"
    if c in (96, 99):
        return "Thunderstorm with hail"
    return "Mixed conditions"


def _crop_tool_suffix(crop: str) -> str:
    return (crop or "").strip().lower().replace(" ", "_")


def weather_tool_cache_key(lat: float, lng: float, crop: str) -> str:
    """Redis key for 7‑day agronomy weather tool payloads."""
    return f"weather_tool:{ck.normalize_geo(lat, lng)}:{_crop_tool_suffix(crop)}"


async def get_weather_widget(lat: float, lng: float, settings: Optional[Settings] = None) -> Dict[str, Any]:
    """Phone-style home widget: live current block + multi-day daily forecast."""
    settings = settings or get_settings()
    if settings.redis_configured:
        r = get_redis_for_request(settings)
        if r:
            cached = await json_get_maybe(r, ck.weather_key(lat, lng))
            if isinstance(cached, dict) and cached.get("current"):
                return cached

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lng,
        "current": ",".join(
            [
                "temperature_2m",
                "apparent_temperature",
                "relative_humidity_2m",
                "weather_code",
                "wind_speed_10m",
                "precipitation",
            ]
        ),
        "daily": ",".join(
            [
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "precipitation_probability_max",
                "weather_code",
            ]
        ),
        "timezone": "Asia/Kolkata",
        "forecast_days": 5,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

    cur = data.get("current") or {}
    cur_code = cur.get("weather_code")
    current = {
        "temperature_c": float(cur.get("temperature_2m") or 0.0),
        "feels_like_c": float(cur.get("apparent_temperature") or cur.get("temperature_2m") or 0.0),
        "humidity_pct": float(cur.get("relative_humidity_2m") or 0.0),
        "wind_speed_kmh": float(cur.get("wind_speed_10m") or 0.0),
        "precipitation_mm": float(cur.get("precipitation") or 0.0),
        "weather_code": int(cur_code) if cur_code is not None else 0,
        "condition": wmo_weather_condition(int(cur_code) if cur_code is not None else None),
    }

    daily = data.get("daily") or {}
    dates: List[str] = list(daily.get("time") or [])
    tmax = daily.get("temperature_2m_max") or []
    tmin = daily.get("temperature_2m_min") or []
    prec_sum = daily.get("precipitation_sum") or []
    prec_prob = daily.get("precipitation_probability_max") or []
    codes = daily.get("weather_code") or []

    forecast: List[Dict[str, Any]] = []
    for i in range(min(5, len(dates))):
        wc = codes[i] if i < len(codes) else None
        forecast.append(
            {
                "date": dates[i],
                "temp_max_c": float(tmax[i]) if i < len(tmax) and tmax[i] is not None else 0.0,
                "temp_min_c": float(tmin[i]) if i < len(tmin) and tmin[i] is not None else 0.0,
                "precipitation_mm": round(float(prec_sum[i]), 2) if i < len(prec_sum) and prec_sum[i] is not None else 0.0,
                "precip_probability_pct": float(prec_prob[i]) if i < len(prec_prob) and prec_prob[i] is not None else 0.0,
                "weather_code": int(wc) if wc is not None else 0,
                "condition": wmo_weather_condition(int(wc) if wc is not None else None),
            }
        )

    result = {"current": current, "forecast": forecast}
    if settings.redis_configured:
        r = get_redis_for_request(settings)
        if r:
            try:
                await json_setex(
                    r,
                    ck.weather_key(lat, lng),
                    ck.ttl_weather(settings),
                    result,
                )
            except Exception:
                logger.debug("Redis widget write skipped", exc_info=True)
    return result


async def get_weather(lat: float, lng: float, crop: str, settings: Optional[Settings] = None) -> Dict[str, Any]:
    """Fetch 7-day outlook + rain risk + irrigation hint."""
    settings = settings or get_settings()
    if settings.redis_configured:
        r = get_redis_for_request(settings)
        if r:
            tk = weather_tool_cache_key(lat, lng, crop)
            hit = await json_get_maybe(r, tk)
            if isinstance(hit, dict) and hit.get("outlook"):
                return hit

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

    out = {
        "outlook": outlook,
        "rain_risk": rain_risk,
        "irrigation_hint": irrigation_hint,
        "urgency": urgency,
        "source": "open_meteo",
    }
    if settings.redis_configured:
        r_rd = get_redis_for_request(settings)
        if r_rd:
            try:
                await json_setex(
                    r_rd,
                    weather_tool_cache_key(lat, lng, crop),
                    ck.ttl_weather(settings),
                    out,
                )
            except Exception:
                logger.debug("Redis tool weather write skipped", exc_info=True)

    return out
