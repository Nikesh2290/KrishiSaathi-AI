"""Mandi prices from bundled CSV + simple trend label."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


def _csv_path() -> Path:
    return Path(__file__).resolve().parents[2] / "offline" / "data" / "mandi_prices.csv"


def get_prices(crop: str, district: str) -> Dict[str, Any]:
    path = _csv_path()
    if not path.exists():
        return {"error": "mandi_prices.csv missing", "spot_price_inr": None, "trend": "unknown"}
    rows: List[Dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("crop", "").lower() == crop.lower() and row.get("district", "").lower() == district.lower():
                rows.append(row)
    if not rows:
        return {
            "spot_price_inr": None,
            "trend": "unknown",
            "note": f"No rows for {crop}/{district} in offline data.",
        }
    prices = [float(x["price_inr"]) for x in rows if x.get("price_inr")]
    spot = prices[-1]
    if len(prices) >= 3:
        recent = mean(prices[-3:])
        older = mean(prices[-6:-3]) if len(prices) >= 6 else recent
        if recent > older * 1.02:
            trend = "up"
        elif recent < older * 0.98:
            trend = "down"
        else:
            trend = "flat"
    else:
        trend = "flat"
    hint = "Consider holding if prices are rising; watch mandi arrivals." if trend == "up" else "Prices stable or soft — plan logistics."
    return {
        "spot_price_inr": spot,
        "unit": rows[-1].get("unit", "quintal"),
        "trend": trend,
        "mandi": rows[-1].get("mandi", ""),
        "buy_sell_hint": hint,
        "source": "offline_csv",
    }
