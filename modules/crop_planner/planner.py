"""Rule-based crop recommendation + optional LLM rationale."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from agent.gemma_client import generate
from config.settings import Settings, get_settings


def _load_calendar() -> Dict[str, Any]:
    path = Path(__file__).resolve().parents[2] / "offline" / "data" / "crop_calendar.json"
    if not path.exists():
        return {"regions": {}, "crops": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def recommend(
    soil: str,
    state: str,
    season: str,
    water_source: str,
    prefer_local: bool,
    settings: Settings | None = None,
) -> Dict[str, Any]:
    settings = settings or get_settings()
    cal = _load_calendar()
    crops = cal.get("crops") or {}
    scores: List[Dict[str, Any]] = []
    for name, meta in crops.items():
        soil_fit = 1.0 if soil.lower() in [s.lower() for s in meta.get("soils", [])] else 0.6
        season_fit = 1.0 if season.lower() in [s.lower() for s in meta.get("seasons", [])] else 0.5
        water_need = meta.get("water_need", 2)  # 1-3
        if "tube" in water_source.lower() or "well" in water_source.lower():
            water_fit = 1.0 if water_need <= 2 else 0.7
        else:
            water_fit = 1.0 if water_need >= 2 else 0.8
        region_bonus = 1.0
        for st, mult in (meta.get("state_bonus") or {}).items():
            if st.lower() == state.lower():
                region_bonus = float(mult)
                break
        score = round(soil_fit * season_fit * water_fit * region_bonus * 100, 1)
        scores.append({"crop": name, "score": score, "meta": meta})
    scores.sort(key=lambda x: -x["score"])
    top = scores[:3]

    # LLM rationale (async caller should use recommend_async); sync returns placeholder rationale
    return {
        "top_crops": top,
        "rationale": "Scores combine soil fit, season, and water availability for your state.",
    }


async def recommend_async(
    soil: str,
    state: str,
    season: str,
    water_source: str,
    prefer_local: bool,
    settings: Settings | None = None,
) -> Dict[str, Any]:
    base = recommend(soil, state, season, water_source, prefer_local, settings)
    settings = settings or get_settings()
    msgs = [
        {
            "role": "system",
            "content": "You are an agronomist. In 3-4 sentences, explain the top crop choices for an Indian farmer. Use simple language.",
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "state": state,
                    "soil": soil,
                    "season": season,
                    "water": water_source,
                    "top_crops": base["top_crops"],
                },
                ensure_ascii=False,
            ),
        },
    ]
    try:
        rationale = await generate(msgs, prefer_local=prefer_local, settings=settings)
    except Exception:
        rationale = base["rationale"]
    base["rationale"] = rationale
    return base
