"""KCC / PM-Fasal style numbers in code; LLM only narrates."""

from __future__ import annotations

import json
from typing import Any, Dict

from agent.gemma_client import generate
from config.settings import Settings, get_settings
from models.farmer import FarmerTwin


def _kcc_limit(acres: float) -> float:
    return min(acres * 50000.0, 3_000_000.0)


def _pmfb_premium(sum_insured: float, season: str) -> float:
    rate = 0.015 if season.lower() == "rabi" else 0.02
    return round(sum_insured * rate, 2)


def advise_sync(twin: FarmerTwin, season: str = "rabi") -> Dict[str, Any]:
    acres = float(twin.land.total_acres or 0)
    kcc = _kcc_limit(acres)
    sum_insured = acres * 40000.0  # illustrative sum insured per acre
    prem = _pmfb_premium(sum_insured, season)
    return {
        "kcc_suggested_limit_inr": kcc,
        "pmfb_sum_insured_inr": round(sum_insured, 2),
        "pmfb_estimated_premium_inr": prem,
        "numbers_source": "rule_engine",
    }


async def advise(
    twin: FarmerTwin,
    prefer_local: bool,
    season: str = "rabi",
    settings: Settings | None = None,
    *,
    voice_mode: bool = False,
) -> Dict[str, Any]:
    settings = settings or get_settings()
    numbers = advise_sync(twin, season=season)
    name_s = (twin.name.strip() if twin.name else "") or "the farmer"
    msgs = [
        {
            "role": "system",
            "content": (
                f"Explain these financial numbers to {name_s} in simple Hindi or English. "
                "Do NOT invent new numbers; only narrate what is given."
            ),
        },
        {"role": "user", "content": json.dumps(numbers, ensure_ascii=False)},
    ]
    narrative = await generate(
        msgs, prefer_local=prefer_local, settings=settings, voice_mode=voice_mode
    )
    numbers["narrative"] = narrative
    return numbers
