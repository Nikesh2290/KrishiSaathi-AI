"""Redis key patterns and TTL resolution from settings."""

from __future__ import annotations

from config.settings import Settings


def normalize_geo(lat: float, lng: float) -> str:
    """Stable key suffix for coarse geo (reduces cardinality)."""
    return f"{lat:.4f}:{lng:.4f}"


def twin_key(farmer_id: str) -> str:
    return f"twin:{farmer_id}"


def context_packet_key(farmer_id: str, conversation_id: str) -> str:
    return f"context_packet:{farmer_id}:{conversation_id}"


def session_key(farmer_id: str, conversation_id: str) -> str:
    return f"session:{farmer_id}:{conversation_id}"


def weather_key(lat: float, lng: float) -> str:
    return f"weather:{normalize_geo(lat, lng)}"


def mandi_key(state: str, district: str) -> str:
    return f"mandi:{state.strip().lower()}:{district.strip().lower()}"


def schemes_index_key() -> str:
    return "schemes:index"


def warmup_meta_key(name: str) -> str:
    return f"warmup:{name}"


def ttl_twin(settings: Settings) -> int:
    return int(settings.redis_twin_ttl_seconds)


def ttl_context_packet(settings: Settings) -> int:
    return int(settings.redis_context_packet_ttl_seconds)


def ttl_session(settings: Settings) -> int:
    return int(settings.redis_session_ttl_seconds)


def ttl_weather(settings: Settings) -> int:
    return int(settings.redis_weather_ttl_seconds)


def ttl_mandi(settings: Settings) -> int:
    return int(settings.redis_mandi_ttl_seconds)


def ttl_schemes_index(settings: Settings) -> int:
    return int(settings.redis_schemes_index_ttl_seconds)
