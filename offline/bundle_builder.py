"""Build a district-scoped offline bundle from seed data."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent / "data"


def _load_schemes() -> List[Dict[str, Any]]:
    p = ROOT / "scheme_index.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def _load_crop_calendar() -> Dict[str, Any]:
    p = ROOT / "crop_calendar.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _load_weather_history(district: str) -> List[Dict[str, Any]]:
    p = ROOT / "weather_history.json"
    if not p.exists():
        return []
    rows = json.loads(p.read_text(encoding="utf-8"))
    return [r for r in rows if r.get("district") == district]


def _load_mandi_prices(district: str) -> List[Dict[str, Any]]:
    p = ROOT / "mandi_prices.csv"
    if not p.exists():
        return []
    out: List[Dict[str, Any]] = []
    with p.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("district") == district:
                try:
                    row["price_inr"] = float(row["price_inr"])
                except (TypeError, ValueError):
                    pass
                out.append(row)
    out.sort(key=lambda r: r.get("date", ""), reverse=True)
    return out[:120]


def build_bundle(state: str, district: str) -> Dict[str, Any]:
    data = {
        "schemes": _load_schemes(),
        "mandi_prices": _load_mandi_prices(district),
        "crop_calendar": _load_crop_calendar(),
        "weather_history": _load_weather_history(district),
    }
    key = f"{state}|{district}|{json.dumps(data, sort_keys=True, default=str)}".encode("utf-8")
    digest = hashlib.sha1(key).hexdigest()[:12]
    version = f"{state.lower()}-{district.lower()}-{digest}"
    return {
        "bundle_version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "district": district,
        "state": state,
        "data": data,
        "ttl_hours": 24,
    }


def build_gzip_bundle(state: str, district: str) -> Tuple[bytes, str]:
    bundle = build_bundle(state, district)
    raw = json.dumps(bundle, ensure_ascii=False, default=str).encode("utf-8")
    return gzip.compress(raw), bundle["bundle_version"]
