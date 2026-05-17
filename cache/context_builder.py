"""Build / read Redis context packets (farmer twin + rolling chat turns)."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from agent.history import build_chat_history_context
from cache import cache_keys, redis_client
from config.settings import Settings, get_settings
from models.farmer import FarmerTwin

logger = logging.getLogger(__name__)

MAX_SESSION_TURNS = 50
PLANNER_HISTORY_TURNS = 3


def _turns_from_session(session_data: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if not session_data:
        return []
    return session_data[-PLANNER_HISTORY_TURNS:] if len(session_data) > PLANNER_HISTORY_TURNS else session_data


async def cache_farmer_twin(r: Any, twin: FarmerTwin, settings: Settings) -> None:
    key = cache_keys.twin_key(twin.farmer_id)
    await redis_client.json_setex(
        r,
        key,
        cache_keys.ttl_twin(settings),
        json.loads(twin.model_dump_json()),
    )


async def load_farmer_twin_dict(r: Any, farmer_id: str) -> Optional[Dict[str, Any]]:
    raw = await redis_client.json_get_maybe(r, cache_keys.twin_key(farmer_id))
    if isinstance(raw, dict):
        return raw
    return None


async def load_route_prefetch(
    r: Any,
    farmer_id: str,
    conversation_id: str,
    settings: Settings,
) -> Optional[Dict[str, Any]]:
    """Return keys for AgentState when context_packet hits Redis."""
    key = cache_keys.context_packet_key(farmer_id, conversation_id)
    packet = await redis_client.json_get_maybe(r, key)
    if not packet or not isinstance(packet, dict):
        return None

    twin: Optional[FarmerTwin] = None
    t = packet.get("farmer_twin")
    try:
        if isinstance(t, dict):
            twin = FarmerTwin.model_validate(t)
    except Exception as e:
        logger.debug("Twin in packet invalid: %s", e)

    hist = packet.get("chat_history")
    if hist is None and isinstance(packet.get("session_turns"), list):
        hist = build_chat_history_context(_turns_from_session(packet["session_turns"]))

    if not isinstance(hist, list):
        hist = []

    return {"prefetched_farmer_twin": twin, "prefetched_chat_history": hist, "redis_context_hit": True}


async def rebuild_context_packet(
    r: Any,
    farmer_id: str,
    conversation_id: str,
    twin_fallback: Optional[FarmerTwin],
    settings: Settings,
) -> None:
    twin_dict: Optional[Dict[str, Any]] = None

    redis_t = await load_farmer_twin_dict(r, farmer_id)
    if redis_t:
        twin_dict = redis_t
    elif twin_fallback is not None:
        twin_dict = json.loads(twin_fallback.model_dump_json())

    sess = await redis_client.json_get_maybe(r, cache_keys.session_key(farmer_id, conversation_id))
    session_turns = sess if isinstance(sess, list) else []

    planner_hist = build_chat_history_context(_turns_from_session(session_turns))
    packet = {
        "farmer_twin": twin_dict,
        "chat_history": planner_hist,
        "session_turns": session_turns[-10:],
    }
    await redis_client.json_setex(
        r,
        cache_keys.context_packet_key(farmer_id, conversation_id),
        cache_keys.ttl_context_packet(settings),
        packet,
    )


async def append_turn_update_redis(
    farmer_id: str,
    conversation_id: str,
    query_text: str,
    response: str,
    settings: Optional[Settings] = None,
) -> None:
    settings = settings or get_settings()
    if not settings.redis_configured:
        return
    r = redis_client.redis_from_settings(settings)

    sess_key = cache_keys.session_key(farmer_id, conversation_id)
    session_turns = await redis_client.json_get_maybe(r, sess_key)
    if not isinstance(session_turns, list):
        session_turns = []

    session_turns.append({"query_text": query_text or "", "response": (response or "")[:3500]})
    if len(session_turns) > MAX_SESSION_TURNS:
        session_turns = session_turns[-MAX_SESSION_TURNS:]

    await redis_client.json_setex(
        r,
        sess_key,
        cache_keys.ttl_session(settings),
        session_turns,
    )

    twin_fb: Optional[FarmerTwin] = None
    tdict = await load_farmer_twin_dict(r, farmer_id)
    if tdict:
        try:
            twin_fb = FarmerTwin.model_validate(tdict)
        except Exception:
            twin_fb = None

    await rebuild_context_packet(r, farmer_id, conversation_id, twin_fb, settings)
