"""Unified LLM client: Ollama (local) + Google AI Studio (Gemini API) fallback."""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import AsyncIterator
from typing import List, Optional, Sequence, Union

import httpx
from google import genai as _genai
from google.genai import types as _gtypes
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


def _ollama_messages_payload(
    settings: Settings,
    messages: Sequence[Union[dict, BaseMessage]],
    images: Optional[List[str]] = None,
    stream: bool = False,
) -> tuple[str, dict]:
    """Build POST body for Ollama /api/chat."""
    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    o_msgs: List[dict] = []
    for m in messages:
        if isinstance(m, BaseMessage):
            role = "assistant" if isinstance(m, AIMessage) else ("system" if isinstance(m, SystemMessage) else "user")
            o_msgs.append({"role": role, "content": m.content})
        else:
            o_msgs.append({"role": m["role"], "content": m["content"]})
    if images:
        last = o_msgs[-1]
        last["images"] = images
    payload = {"model": settings.ollama_model, "messages": o_msgs, "stream": stream}
    return url, payload


async def _ollama_chat(
    settings: Settings,
    messages: Sequence[Union[dict, BaseMessage]],
    images: Optional[List[str]] = None,
) -> str:
    url, payload = _ollama_messages_payload(settings, messages, images)
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "") or ""


async def _ollama_chat_stream(
    settings: Settings,
    messages: Sequence[Union[dict, BaseMessage]],
    images: Optional[List[str]] = None,
) -> AsyncIterator[str]:
    """Stream incremental assistant text chunks from Ollama /api/chat (NDJSON)."""
    url, payload = _ollama_messages_payload(settings, messages, images, stream=True)
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", url, json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                line = (line or "").strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("Ollama stream skip non-json line: %s", line[:80])
                    continue
                msg = data.get("message") or {}
                chunk = msg.get("content") or ""
                if chunk:
                    yield chunk
                if data.get("done"):
                    break


def _studio_llm(settings: Settings, heavy: bool = False) -> ChatGoogleGenerativeAI:
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_AI_STUDIO_KEY not set")
    model = settings.ai_studio_model_heavy if heavy else settings.ai_studio_model
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=settings.google_api_key,
        temperature=0.2,
    )


async def generate(
    messages: Sequence[Union[dict, BaseMessage]],
    prefer_local: bool = False,
    settings: Optional[Settings] = None,
    heavy: bool = False,
) -> str:
    settings = settings or get_settings()
    if prefer_local:
        try:
            return await _ollama_chat(settings, messages)
        except Exception as e:
            logger.warning("Ollama generate failed, falling back to AI Studio: %s", e)
    llm = _studio_llm(settings, heavy=heavy)
    lc = _to_lc_messages(messages)
    resp = await llm.ainvoke(lc)
    return str(resp.content)


async def generate_stream(
    messages: Sequence[Union[dict, BaseMessage]],
    prefer_local: bool = False,
    settings: Optional[Settings] = None,
    heavy: bool = False,
) -> AsyncIterator[str]:
    """Yield text chunks from Ollama (stream) or Gemini (LangChain astream).

    Mirrors ``generate()`` routing: try local stream first when ``prefer_local``,
    then fall back to AI Studio streaming.
    """
    settings = settings or get_settings()
    if prefer_local:
        try:
            async for chunk in _ollama_chat_stream(settings, messages):
                yield chunk
            return
        except Exception as e:
            logger.warning("Ollama generate_stream failed, falling back to AI Studio: %s", e)
    llm = _studio_llm(settings, heavy=heavy)
    lc = _to_lc_messages(messages)
    async for chunk in llm.astream(lc):
        raw = getattr(chunk, "content", None)
        if raw is None:
            continue
        if isinstance(raw, str):
            if raw:
                yield raw
        elif isinstance(raw, list):
            # multimodal fragments; synthesize-only path is text-only
            for item in raw:
                if isinstance(item, str):
                    if item:
                        yield item
                elif isinstance(item, dict) and item.get("type") == "text":
                    t = item.get("text") or ""
                    if t:
                        yield t


async def generate_with_vision(
    system_prompt: str,
    user_text: str,
    image_b64: str,
    prefer_local: bool = True,
    settings: Optional[Settings] = None,
) -> str:
    """Multimodal: image + text.

    Ollama path (prefer_local=True): sends base64 in Ollama /api/chat format.
    AI Studio path: uses google-genai SDK directly with Part.from_bytes — this is
    the only reliable way to pass an image; LangChain's image_url/data-URI wrappers
    are inconsistent across langchain-google-genai versions and should NOT be used.
    """
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

    # Decode and validate before making the API call.
    try:
        raw_bytes = base64.b64decode(image_b64, validate=True)
    except Exception as e:
        raise ValueError("Invalid base64 image") from e

    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_AI_STUDIO_KEY not set")

    logger.info(
        "vision → AI Studio model=%s image_bytes=%d",
        settings.ai_studio_model,
        len(raw_bytes),
    )

    client = _genai.Client(api_key=settings.google_api_key)
    response = await client.aio.models.generate_content(
        model=settings.ai_studio_model,
        contents=[
            _gtypes.Part.from_bytes(data=raw_bytes, mime_type="image/jpeg"),
            _gtypes.Part(text=user_text),
        ],
        config=_gtypes.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
        ),
    )
    # Collect only non-thinking text parts. Thinking models return parts with
    # thought=True which must be excluded before JSON parsing.
    text = ""
    for candidate in response.candidates or []:
        for part in (candidate.content.parts if candidate.content else []):
            if getattr(part, "thought", False):
                continue
            if part.text:
                text += part.text
    logger.debug("vision response (first 300): %s", text[:300])
    return text


async def ollama_healthy(settings: Optional[Settings] = None) -> bool:
    settings = settings or get_settings()
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags")
            return r.status_code == 200
    except Exception:
        return False
