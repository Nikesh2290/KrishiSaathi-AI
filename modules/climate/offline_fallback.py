"""Offline weather from bundled JSON history."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


_HIST = Path(__file__).resolve().parents[2] / "offline" / "data" / "weather_history.json"


def _load() -> List[Dict[str, Any]]:
    if not _HIST.exists():
        return []
    try:
        data = json.loads(_HIST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def offline_weather(district_key: str, crop: str) -> Dict[str, Any]:
    """Historical monthly averages; district_key matched case-insensitively."""
    rows = _load()
    if not rows:
        return {
            "outlook": {"note": "No offline weather file yet."},
            "rain_risk": "unknown",
            "irrigation_hint": "Connect to internet or sync offline data.",
            "urgency": "low",
            "disclaimer": "Using cached offline data. Last updated: never",
        }

    key = (district_key or "").lower()
    matching = [r for r in rows if str(r.get("district", "")).lower() == key]
    disclaimer = "Using cached offline data."

    if not matching:
        return {
            "outlook": {"note": "District not in offline set."},
            "rain_risk": "unknown",
            "irrigation_hint": "Use district with offline coverage.",
            "urgency": "low",
            "disclaimer": disclaimer,
        }

    temps = [float(r.get("avg_temp_c", 0) or 0) for r in matching]
    rains = [float(r.get("avg_rain_mm", 0) or 0) for r in matching]
    avg_temp = sum(temps) / len(temps) if temps else 0.0
    avg_rain = sum(rains) / len(rains) if rains else 0.0
    rain_risk = "high" if avg_rain > 80 else ("medium" if avg_rain > 40 else "low")
    return {
        "outlook": {"avg_temp_c": avg_temp, "avg_rain_mm_month": avg_rain, "crop": crop},
        "rain_risk": rain_risk,
        "irrigation_hint": "Based on historical averages for your district.",
        "urgency": "low",
        "disclaimer": disclaimer,
        "source": "offline_json",
    }
