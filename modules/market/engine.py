"""Mandi prices: local SQLite (OGD sync) first, live OGD fallback, then bundled CSV."""

from __future__ import annotations

import asyncio
import csv
import logging
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from config.settings import Settings, get_settings
from db.sqlite_client import get_mandi_prices, upsert_mandi_prices_bulk
from modules.market import ogd_client

from cache import cache_keys as ck
from cache.redis_client import get_redis_for_request, json_get_maybe

logger = logging.getLogger(__name__)


def _csv_path() -> Path:
    return Path(__file__).resolve().parents[2] / "offline" / "data" / "mandi_prices.csv"


def _aggregate_rows(rows: List[Dict[str, Any]], source: str) -> Dict[str, Any]:
    """Build tool response from rows with ``modal_price`` (newest rows first)."""
    if not rows:
        return {
            "spot_price_inr": None,
            "trend": "unknown",
            "note": "No mandi rows to aggregate.",
            "source": source,
        }
    prices = [float(r["modal_price"]) for r in rows if r.get("modal_price") is not None]
    if not prices:
        return {
            "spot_price_inr": None,
            "trend": "unknown",
            "note": "No modal prices in rows.",
            "source": source,
        }
    spot = prices[0]
    if len(prices) >= 3:
        recent = mean(prices[:3])
        older = mean(prices[3:6]) if len(prices) >= 6 else recent
        if recent > older * 1.02:
            trend = "up"
        elif recent < older * 0.98:
            trend = "down"
        else:
            trend = "flat"
    else:
        trend = "flat"
    hint = (
        "Consider holding if prices are rising; watch mandi arrivals."
        if trend == "up"
        else "Prices stable or soft — plan logistics."
    )
    last = rows[0]
    return {
        "spot_price_inr": spot,
        "unit": "quintal",
        "trend": trend,
        "mandi": str(last.get("market") or ""),
        "variety": str(last.get("variety") or ""),
        "price_date": str(last.get("price_date") or ""),
        "buy_sell_hint": hint,
        "source": source,
    }


def _get_prices_csv(crop: str, district: str) -> Dict[str, Any]:
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


async def get_prices(
    crop: str,
    district: str,
    settings: Optional[Settings] = None,
    *,
    state: str,
) -> Dict[str, Any]:
    """Resolve mandi prices: local DB → OGD HTTP → CSV seed."""
    settings = settings or get_settings()
    crop_s = (crop or "").strip() or "wheat"
    dist_s = (district or "").strip() or "Ludhiana"
    state_s = (state or "").strip()

    if settings.redis_configured:
        redis_c = get_redis_for_request(settings)
        if redis_c:
            raw = await json_get_maybe(redis_c, ck.mandi_key(state_s, dist_s))
            if isinstance(raw, dict):
                recs = raw.get("records") or []
                matching = [
                    r
                    for r in recs
                    if str(r.get("commodity", "") or "").strip().lower() == crop_s.lower()
                ]
                if matching:
                    return _aggregate_rows(matching, "redis_cache")

    local_rows = await get_mandi_prices(crop_s, dist_s, settings)
    if local_rows:
        return _aggregate_rows(local_rows, "local_db")

    if settings.ogd_api_key and settings.ogd_api_key.strip():
        try:
            records = await ogd_client.fetch_mandi_prices(
                state_s or None,
                dist_s,
                crop_s,
                settings.ogd_api_key.strip(),
                timeout=float(settings.market_tool_timeout_seconds),
            )
            if records:
                use_state = state_s or str(records[0].get("state") or "").strip()
                if not use_state:
                    use_state = state_s or "Unknown"
                await upsert_mandi_prices_bulk(
                    use_state,
                    dist_s,
                    records,
                    settings.mandi_price_ttl_seconds,
                    settings,
                    commodity_filter=crop_s,
                )
                local_rows = await get_mandi_prices(crop_s, dist_s, settings)
                if local_rows:
                    return _aggregate_rows(local_rows, "ogd_live")
        except Exception as e:
            logger.warning("OGD mandi fetch failed, using CSV fallback: %s", e)

    out = await asyncio.to_thread(_get_prices_csv, crop_s, dist_s)
    if "source" not in out:
        out["source"] = "offline_csv"
    return out
