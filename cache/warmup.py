"""Pre-warm Upstash Redis (weather, mandi, schemes metadata) + Upstash Vector embeddings."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cache import cache_keys, gemini_embed, redis_client, vector_client
from config.settings import Settings, get_settings
from modules.climate import engine as climate_engine
from modules.market import ogd_client

logger = logging.getLogger(__name__)


def _scheme_path() -> Path:
    return Path(__file__).resolve().parents[1] / "offline" / "data" / "scheme_index.json"


def _parse_states_districts(settings: Settings) -> List[Tuple[Optional[str], str]]:
    states = [s.strip() for s in settings.warmup_states.split(",") if s.strip()]
    districts = [d.strip() for d in settings.warmup_districts.split(",") if d.strip()]
    out: List[Tuple[Optional[str], str]] = []
    if states and districts and len(states) == len(districts):
        out.extend(zip(states, districts))  # type: ignore[arg-type]
    elif districts:
        for d in districts:
            st = states[0] if len(states) == 1 else (states[len(out) % len(states)] if states else None)
            out.append((st, d))
    elif states:
        logger.warning("WARMUP_STATES set without WARMUP_DISTRICTS; skipping mandi warm-up")
    return out


def _parse_coordinates(settings: Settings) -> List[Tuple[float, float]]:
    coords: List[Tuple[float, float]] = []
    raw = (settings.warmup_coordinates or "").strip()
    if not raw:
        return coords
    for part in raw.split("|"):
        p = part.strip()
        if not p or ":" not in p:
            continue
        a, b = p.split(":", 1)
        try:
            coords.append((float(a.strip()), float(b.strip())))
        except ValueError:
            logger.warning("Bad WARMUP_COORDINATES fragment: %r", part)
    return coords


async def warm_weather_for_coords(settings: Settings, r: Any) -> int:
    n = 0
    ttl = cache_keys.ttl_weather(settings)
    for lat, lng in _parse_coordinates(settings):
        try:
            widget = await climate_engine.get_weather_widget(lat, lng)
            key = cache_keys.weather_key(lat, lng)
            await redis_client.json_setex(r, key, ttl, widget)
            n += 1
        except Exception as e:
            logger.warning("Weather warm failed for (%s,%s): %s", lat, lng, e)
    return n


async def warm_mandi_districts(settings: Settings, r: Any) -> int:
    if not settings.ogd_api_key.strip():
        logger.info("Skipping mandi Redis warm-up: OGD_API_KEY not set")
        return 0
    pairs = _parse_states_districts(settings)
    n = 0
    ttl = cache_keys.ttl_mandi(settings)
    for state, district in pairs:
        try:
            rows = await ogd_client.fetch_mandi_prices(
                state or None,
                district,
                None,
                settings.ogd_api_key.strip(),
                timeout=float(settings.market_tool_timeout_seconds),
            )
            key = cache_keys.mandi_key(state or "", district)
            await redis_client.json_setex(
                r,
                key,
                ttl,
                {"records": rows, "saved_at": int(time.time()), "district": district, "state": state or ""},
            )
            n += 1
        except Exception as e:
            logger.warning("Mandi warm failed for %s/%s: %s", state, district, e)
    return n


async def warm_schemes_index(settings: Settings, r: Any) -> None:
    path = _scheme_path()
    if not path.exists():
        return
    schemes = json.loads(path.read_text(encoding="utf-8"))
    blob = json.dumps(schemes, sort_keys=True, default=str).encode()
    digest = hashlib.sha1(blob).hexdigest()[:16]
    await redis_client.json_setex(
        r,
        cache_keys.schemes_index_key(),
        cache_keys.ttl_schemes_index(settings),
        {"digest": digest, "count": len(schemes), "updated_at": int(time.time())},
    )


async def warm_scheme_vectors(settings: Settings) -> int:
    """Upsert scheme chunks into Upstash Vector using Gemini embeddings."""
    ix = vector_client.get_async_vector_index(settings)
    path = _scheme_path()
    if ix is None or not path.exists() or not settings.google_api_key.strip():
        return 0

    schemes = json.loads(path.read_text(encoding="utf-8"))
    texts: List[str] = []
    ids: List[str] = []
    meta_json: List[str] = []
    for i, s in enumerate(schemes):
        sid = str(s.get("id") or f"scheme_{i}")
        text = "\n".join(
            [
                s.get("name", ""),
                s.get("eligibility", ""),
                s.get("benefits", ""),
                s.get("how_to_apply", ""),
                " ".join(s.get("keywords", [])),
            ]
        )
        texts.append(text)
        ids.append(sid)
        meta_json.append(json.dumps(s, ensure_ascii=False))

    batch = 12
    done = 0
    for i in range(0, len(texts), batch):
        chunk_t = texts[i : i + batch]
        embeddings = await asyncio.to_thread(gemini_embed.embed_documents_sync, settings, chunk_t)
        chunk_ids = ids[i : i + batch]
        chunk_meta = meta_json[i : i + batch]
        vecs = [
            {"id": chunk_ids[j], "vector": embeddings[j], "metadata": {"json": chunk_meta[j]}}
            for j in range(len(chunk_t))
            if j < len(embeddings)
        ]
        if vecs:
            await ix.upsert(vectors=vecs)
            done += len(vecs)
    return done


async def run_warmup(
    settings: Optional[Settings] = None,
    *,
    scopes: Optional[List[str]] = None,
    include_scheme_vectors: bool = True,
) -> Dict[str, Any]:
    settings = settings or get_settings()
    if not settings.redis_configured:
        return {"ok": False, "reason": "redis_not_configured"}

    r = redis_client.redis_from_settings(settings)

    scopes = scopes or ["weather", "mandi", "schemes_index"]

    stats: Dict[str, Any] = {"scopes": scopes, "counts": {}, "vectors_upserted": 0}
    try:
        if "weather" in scopes:
            stats["counts"]["weather_coords"] = await warm_weather_for_coords(settings, r)
        if "mandi" in scopes:
            stats["counts"]["mandi_districts"] = await warm_mandi_districts(settings, r)
        if "schemes_index" in scopes:
            await warm_schemes_index(settings, r)
            stats["counts"]["schemes_index"] = True
        if include_scheme_vectors and "schemes_vector" in scopes:
            stats["vectors_upserted"] = await warm_scheme_vectors(settings)

        await r.set(cache_keys.warmup_meta_key("last_run"), str(int(time.time())))
    except Exception:
        logger.exception("run_warmup failed")
        raise
    stats["ok"] = True
    return stats


def scopes_for_cron(freq: str) -> List[str]:
    if freq == "frequent":
        return ["weather", "mandi", "schemes_index"]
    if freq == "daily_vectors":
        return ["schemes_vector"]
    return ["weather", "mandi", "schemes_index", "schemes_vector"]
