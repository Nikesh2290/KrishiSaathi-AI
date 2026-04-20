"""Unified LLM client: Ollama (local) + Google AI Studio (Gemini API) fallback."""

from __future__ import annotations

import base64
import logging
from typing import Any, List, Optional, Sequence, Union

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


def _to_lc_messages(messages: Sequence[Union[dict, BaseMessage]]) -> List[BaseMessage]:
    out: List[BaseMessage] = []
    for m in messages:
        if isinstance(m, BaseMessage):
            out.append(m)
            continue
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            out.append(SystemMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
        else:
            out.append(HumanMessage(content=content))
    return out


async def _ollama_chat(
    settings: Settings,
    messages: Sequence[Union[dict, BaseMessage]],
    images: Optional[List[str]] = None,
) -> str:
    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    o_msgs: List[dict] = []
    for m in messages:
        if isinstance(m, BaseMessage):
            role = "assistant" if isinstance(m, AIMessage) else ("system" if isinstance(m, SystemMessage) else "user")
            o_msgs.append({"role": role, "content": m.content})
        else:
            o_msgs.append({"role": m["role"], "content": m["content"]})
    if images:
        # Ollama supports images in message for vision models
        last = o_msgs[-1]
        last["images"] = images
    payload = {"model": settings.ollama_model, "messages": o_msgs, "stream": False}
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "") or ""


def _studio_llm(settings: Settings) -> ChatGoogleGenerativeAI:
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_AI_STUDIO_KEY not set")
    return ChatGoogleGenerativeAI(
        model=settings.ai_studio_model,
        google_api_key=settings.google_api_key,
        temperature=0.2,
    )


async def generate(
    messages: Sequence[Union[dict, BaseMessage]],
    prefer_local: bool = True,
    settings: Optional[Settings] = None,
) -> str:
    settings = settings or get_settings()
    if prefer_local:
        try:
            return await _ollama_chat(settings, messages)
        except Exception as e:
            logger.warning("Ollama generate failed, falling back to AI Studio: %s", e)
    llm = _studio_llm(settings)
    lc = _to_lc_messages(messages)
    resp = await llm.ainvoke(lc)
    return str(resp.content)


async def generate_with_vision(
    system_prompt: str,
    user_text: str,
    image_b64: str,
    prefer_local: bool = True,
    settings: Optional[Settings] = None,
) -> str:
    """Multimodal: image + text. Uses AI Studio when Ollama vision unavailable."""
    settings = settings or get_settings()
    if prefer_local:
        try:
            return await _ollama_chat(
                settings,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                images=[image_b64],
            )
        except Exception as e:
            logger.warning("Ollama vision failed, trying AI Studio: %s", e)
    llm = _studio_llm(settings)
    # LangChain Gemini multimodal
    try:
        base64.b64decode(image_b64, validate=True)
    except Exception as e:
        raise ValueError("Invalid base64 image") from e
    msg = HumanMessage(
        content=[
            {"type": "text", "text": f"{system_prompt}\n\n{user_text}"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_b64}},
        ]
    )
    resp = await llm.ainvoke([msg])
    return str(resp.content)


async def ollama_healthy(settings: Optional[Settings] = None) -> bool:
    settings = settings or get_settings()
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags")
            return r.status_code == 200
    except Exception:
        return False
