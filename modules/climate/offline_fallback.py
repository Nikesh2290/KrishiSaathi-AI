"""Offline weather from bundled Parquet (DuckDB)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict

import duckdb

from config.settings import get_settings


def _data_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "offline" / "data"


def offline_weather(district_key: str, crop: str) -> Dict[str, Any]:
    """Historical monthly averages; district_key lowercased slug."""
    path = _data_dir() / "weather_history.parquet"
    if not path.exists():
        return {
            "outlook": {"note": "No offline weather file yet."},
            "rain_risk": "unknown",
            "irrigation_hint": "Connect to internet or sync offline data.",
            "urgency": "low",
            "disclaimer": "Using cached offline data. Last updated: never",
        }
    con = duckdb.connect(database=":memory:")
    rows = con.execute(
        """
        SELECT AVG(avg_temp_c) as t, AVG(avg_rain_mm) as r, MAX(last_updated) as lu
        FROM read_parquet(?)
        WHERE lower(district) = lower(?)
        """,
        [str(path), district_key],
    ).fetchone()
    lu = rows[2] if rows and rows[2] else int(time.time())
    disclaimer = f"Using cached offline data. Last updated: {lu}"
    if not rows or rows[0] is None:
        return {
            "outlook": {"note": "District not in offline set."},
            "rain_risk": "unknown",
            "irrigation_hint": "Use district with offline coverage.",
            "urgency": "low",
            "disclaimer": disclaimer,
        }
    rain_mm = float(rows[1] or 0)
    rain_risk = "high" if rain_mm > 80 else ("medium" if rain_mm > 40 else "low")
    return {
        "outlook": {"avg_temp_c": float(rows[0] or 0), "avg_rain_mm_month": rain_mm, "crop": crop},
        "rain_risk": rain_risk,
        "irrigation_hint": "Based on historical averages for your district.",
        "urgency": "low",
        "disclaimer": disclaimer,
        "source": "offline_parquet",
    }
