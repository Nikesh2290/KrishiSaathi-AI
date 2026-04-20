"""Route between local (Ollama) and cloud (AI Studio) based on connectivity + health."""

from __future__ import annotations

from typing import Literal

from agent.gemma_client import ollama_healthy
from config.settings import Settings, get_settings
from models.request import AgentRequest

Route = Literal["local", "cloud"]


async def resolve_route(req: AgentRequest, settings: Settings | None = None) -> Route:
    settings = settings or get_settings()
    mode = settings.connectivity_mode.lower()
    if mode == "local":
        return "local"
    if mode == "cloud":
        return "cloud"
    # auto
    if req.context.connectivity == "offline":
        return "local"
    ok = await ollama_healthy(settings)
    return "local" if ok else "cloud"


def data_source_for_route(route: Route, req: AgentRequest) -> str:
    if req.context.connectivity == "offline":
        return "offline"
    return "live"
