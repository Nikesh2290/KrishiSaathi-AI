"""Climate module tests."""

from __future__ import annotations

import pytest

from modules.climate import engine as climate_engine


@pytest.mark.asyncio
async def test_get_weather_shape(monkeypatch):
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
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def get(self, url, params=None):
            return FakeResp()

    monkeypatch.setattr("modules.climate.engine.httpx.AsyncClient", lambda timeout=30.0: FakeClient())
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
