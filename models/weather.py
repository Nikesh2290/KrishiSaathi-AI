"""Home-screen weather API response models."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class CurrentWeather(BaseModel):
    temperature_c: float
    feels_like_c: float
    humidity_pct: float
    wind_speed_kmh: float
    precipitation_mm: float
    weather_code: int
    condition: str = Field(..., description="Human-readable summary from WMO weather_code")


class DayForecast(BaseModel):
    date: str
    temp_max_c: float
    temp_min_c: float
    precipitation_mm: float
    precip_probability_pct: float
    weather_code: int
    condition: str


class WeatherResponse(BaseModel):
    farmer_id: str
    lat: float
    lng: float
    current: CurrentWeather
    forecast: List[DayForecast]
    cached: bool
    fetched_at: str = Field(..., description="ISO-8601 when data was fetched from upstream")
    expires_at: str = Field(..., description="ISO-8601 when server cache expires (IST)")
