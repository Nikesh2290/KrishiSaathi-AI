"""Home-screen weather (cached Open-Meteo widget data)."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict
from zoneinfo import ZoneInfo

import httpx
from fastapi import APIRouter, Query

from config.settings import Settings, get_settings
from db.persistence import resolve_farmer_twin
from db.sqlite_client import get_weather_cache, set_weather_cache
from models.errors import ErrorCode, KrishiHTTPException
from models.weather import CurrentWeather, DayForecast, WeatherResponse
from modules.climate import engine as climate_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["weather"])

_IST = ZoneInfo("Asia/Kolkata")


def _iso_ist(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=_IST).isoformat()


async def _get_weather_widget_timeout(
    lat: float,
    lng: float,
    timeout_s: float,
    settings: Settings,
) -> Dict[str, Any]:
    return await asyncio.wait_for(
        climate_engine.get_weather_widget(lat, lng, settings),
        timeout=timeout_s,
    )


@router.get("/weather/{farmer_id}", response_model=WeatherResponse)
async def get_farmer_home_weather(
    farmer_id: str,
    connectivity: str = Query(
        "online",
        description="offline = SQLite twin only",
    ),
    force_refresh: bool = Query(False, description="Bypass cache and re-fetch Open-Meteo"),
) -> WeatherResponse:
    settings = get_settings()
    twin = await resolve_farmer_twin(farmer_id, connectivity, settings)
    if not twin:
        raise KrishiHTTPException(
            status_code=404,
            code=ErrorCode.FARMER_NOT_FOUND,
            message="Farmer twin not found",
        )

    lat_raw, lng_raw = twin.location.lat, twin.location.lng
    if lat_raw is None or lng_raw is None:
        raise KrishiHTTPException(
            status_code=422,
            code=ErrorCode.LOCATION_MISSING,
            message="Farmer twin has no latitude/longitude — set location for weather.",
        )

    lat = float(lat_raw)
    lng = float(lng_raw)
    location_key = f"{lat:.4f},{lng:.4f}"

    ttl = int(settings.weather_cache_ttl_seconds)
    climate_timeout = float(settings.climate_timeout_seconds)

    if not force_refresh:
        hit = await get_weather_cache(location_key, settings)
        if hit is not None:
            inner, fetched_at, expires_at = hit
            return WeatherResponse(
                farmer_id=farmer_id,
                lat=lat,
                lng=lng,
                current=CurrentWeather.model_validate(inner["current"]),
                forecast=[DayForecast.model_validate(d) for d in inner.get("forecast", [])],
                cached=True,
                fetched_at=_iso_ist(fetched_at),
                expires_at=_iso_ist(expires_at),
            )

    try:
        inner = await _get_weather_widget_timeout(lat, lng, climate_timeout, settings)
    except asyncio.TimeoutError:
        logger.warning("Open-Meteo widget timeout farmer_id=%s lat=%s lng=%s", farmer_id, lat, lng)
        raise KrishiHTTPException(
            status_code=503,
            code=ErrorCode.UPSTREAM_UNAVAILABLE,
            message="Weather service timed out",
        )
    except httpx.HTTPError as e:
        logger.warning("Open-Meteo widget HTTP error farmer_id=%s: %s", farmer_id, e)
        raise KrishiHTTPException(
            status_code=503,
            code=ErrorCode.UPSTREAM_UNAVAILABLE,
            message="Weather upstream unavailable",
        )

    fetched_at, expires_at = await set_weather_cache(
        location_key,
        inner,
        ttl_seconds=ttl,
        settings=settings,
    )

    return WeatherResponse(
        farmer_id=farmer_id,
        lat=lat,
        lng=lng,
        current=CurrentWeather.model_validate(inner["current"]),
        forecast=[DayForecast.model_validate(d) for d in inner.get("forecast", [])],
        cached=False,
        fetched_at=_iso_ist(fetched_at),
        expires_at=_iso_ist(expires_at),
    )
