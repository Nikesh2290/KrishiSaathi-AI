"""Climate module tests."""

from __future__ import annotations

import pytest

from modules.climate import engine as climate_engine


@pytest.mark.asyncio
async def test_get_weather_widget_shape(monkeypatch):
    import modules.climate.engine as eng

    monkeypatch.setattr(eng, "_climate_http", None)

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "current": {
                    "temperature_2m": 28.0,
                    "apparent_temperature": 29.0,
                    "relative_humidity_2m": 55.0,
                    "weather_code": 0,
                    "wind_speed_10m": 12.0,
                    "precipitation": 0.0,
                },
                "daily": {
                    "time": [f"2026-05-0{i}" for i in range(1, 6)],
                    "temperature_2m_max": [33.0] * 5,
                    "temperature_2m_min": [18.0] * 5,
                    "precipitation_sum": [0.0] * 5,
                    "precipitation_probability_max": [10.0] * 5,
                    "weather_code": [0, 1, 2, 3, 61],
                },
            }

    class FakeClient:
        async def get(self, url, params=None):
            return FakeResp()

    monkeypatch.setattr(
        eng.httpx,
        "AsyncClient",
        lambda *args, **kwargs: FakeClient(),
    )
    out = await climate_engine.get_weather_widget(30.0, 75.0)
    assert "current" in out
    assert len(out["forecast"]) == 5
    assert out["current"]["condition"] == "Clear sky"
    assert out["forecast"][-1]["condition"] == "Rain"


def test_wmo_weather_condition():
    assert climate_engine.wmo_weather_condition(0) == "Clear sky"
    assert climate_engine.wmo_weather_condition(95) == "Thunderstorm"


@pytest.mark.asyncio
async def test_get_weather_shape(monkeypatch):
    import modules.climate.engine as eng

    monkeypatch.setattr(eng, "_climate_http", None)

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "daily": {
                    "time": ["2026-04-19"] * 7,
                    "temperature_2m_max": [30.0] * 7,
                    "temperature_2m_min": [15.0] * 7,
                    "precipitation_sum": [2.0] * 7,
                    "precipitation_probability_max": [20.0] * 7,
                }
            }

    class FakeClient:
        async def get(self, url, params=None):
            return FakeResp()

    monkeypatch.setattr(
        eng.httpx,
        "AsyncClient",
        lambda *args, **kwargs: FakeClient(),
    )
    out = await climate_engine.get_weather(30.0, 75.0, "wheat")
    assert "rain_risk" in out
    assert out["source"] == "open_meteo"


def test_offline_weather_import():
    from offline.bootstrap_data import bootstrap_all
    from pathlib import Path

    if not Path("offline/data/weather_history.json").exists():
        bootstrap_all()
    from modules.climate.offline_fallback import offline_weather

    out = offline_weather("Ludhiana", "wheat")
    assert "disclaimer" in out or "outlook" in out
