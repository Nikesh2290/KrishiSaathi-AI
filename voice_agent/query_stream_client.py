"""Parse Krishi `POST /api/v1/query/stream` SSE (AI SDK data stream)."""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from voice_agent.timing import VoiceTimeline

logger = logging.getLogger(__name__)

_SSE_LOG_TYPES = frozenset(
    {
        "start",
        "data-stage",
        "data-tool",
        "text-start",
        "text-delta",
        "text-end",
        "data-metadata",
        "finish",
        "error",
    }
)


def _sse_block_to_data_str(block: str) -> str:
    parts: list[str] = []
    for line in block.strip().split("\n"):
        if line.startswith("data:"):
            parts.append(line.removeprefix("data:").strip())
    return "".join(parts).strip()


def _consume_buffer(buf: str) -> tuple[str, str, list[str], dict[str, Any] | None]:
    """Split `buf` on complete SSE frames. Returns (remainder, status, delta_chunks, last_metadata).

    status: "" (continue), "done" ([DONE] seen), "error" (stream error event).
    """
    pieces: list[str] = []
    last_meta: dict[str, Any] | None = None
    out_buf = buf
    status = ""
    while "\n\n" in out_buf:
        block, out_buf = out_buf.split("\n\n", 1)
        data_str = _sse_block_to_data_str(block)
        if not data_str:
            continue
        if data_str == "[DONE]":
            status = "done"
            break
        try:
            obj = json.loads(data_str)
        except json.JSONDecodeError:
            logger.warning("bad sse json: %s", data_str[:200])
            continue
        t = obj.get("type")
        if t == "text-delta" and "delta" in obj:
            pieces.append(str(obj["delta"]))
        if t == "data-metadata":
            inner = obj.get("data")
            if isinstance(inner, dict):
                last_meta = inner
        if t == "error":
            logger.warning("query stream error: %s", obj.get("errorText"))
            status = "error"
            break
    return out_buf, status, pieces, last_meta


def _consume_buffer_events(buf: str) -> tuple[str, str, list[dict[str, Any]]]:
    """Parse complete SSE frames into typed JSON objects.

    Returns (remainder, status, events).
    status: ``\"\"`` | ``\"done\"`` | ``\"error\"``.
    """
    events: list[dict[str, Any]] = []
    out_buf = buf
    status = ""
    while "\n\n" in out_buf:
        block, out_buf = out_buf.split("\n\n", 1)
        data_str = _sse_block_to_data_str(block)
        if not data_str:
            continue
        if data_str == "[DONE]":
            status = "done"
            break
        try:
            obj = json.loads(data_str)
        except json.JSONDecodeError:
            logger.warning("bad sse json: %s", data_str[:200])
            continue
        if not isinstance(obj, dict):
            continue
        events.append(obj)
        if obj.get("type") == "error":
            status = "error"
            break
    return out_buf, status, events


async def collect_from_query_stream(
    client: httpx.AsyncClient,
    *,
    api_base: str,
    payload: dict[str, Any],
) -> tuple[str, dict[str, Any] | None]:
    """POST `/api/v1/query/stream`; return (full_text_from_deltas, last data-metadata inner dict or None)."""
    url = f"{api_base.rstrip('/')}/api/v1/query/stream"
    all_text: list[str] = []
    last_meta: dict[str, Any] | None = None
    buf = ""
    async with client.stream(
        "POST",
        url,
        json=payload,
        headers={"Accept": "text/event-stream"},
        timeout=120.0,
    ) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if line is None:
                continue
            buf += line + "\n"
            buf, status, pieces, meta = _consume_buffer(buf)
            all_text.extend(pieces)
            if meta is not None:
                last_meta = meta
            if status == "done":
                return "".join(all_text).strip(), last_meta
            if status == "error":
                return "", last_meta
        buf, status, pieces, meta = _consume_buffer(buf)
        all_text.extend(pieces)
        if meta is not None:
            last_meta = meta
        if status == "error":
            return "", last_meta
    return "".join(all_text).strip(), last_meta


async def collect_text_from_query_stream(
    client: httpx.AsyncClient,
    *,
    api_base: str,
    payload: dict[str, Any],
) -> str:
    text, _ = await collect_from_query_stream(client, api_base=api_base, payload=payload)
    return text


def _log_sse_event(timeline: VoiceTimeline | None, obj: dict[str, Any], *, first_sse: bool) -> None:
    if timeline is None:
        return
    otype = str(obj.get("type") or "")
    extra: dict[str, Any] = {"sse_type": otype}
    if first_sse:
        extra["first_sse"] = True
    if otype == "data-stage":
        data = obj.get("data") if isinstance(obj.get("data"), dict) else {}
        extra["stage"] = data.get("stage")
        extra["status"] = data.get("status")
    elif otype == "data-tool":
        data = obj.get("data") if isinstance(obj.get("data"), dict) else {}
        extra["tool"] = data.get("tool")
        extra["status"] = data.get("status")
    elif otype == "text-delta":
        delta = str(obj.get("delta") or "")
        extra["delta_chars"] = len(delta)
    elif otype == "error":
        extra["error_text"] = str(obj.get("errorText") or "")[:200]
    if otype in _SSE_LOG_TYPES:
        timeline.mark(f"sse_{otype}", **extra)


async def iter_query_stream_events(
    client: httpx.AsyncClient,
    *,
    api_base: str,
    payload: dict[str, Any],
    timeline: VoiceTimeline | None = None,
    turn_id: str | None = None,
):
    """Yield each SSE JSON event as it arrives (streaming).

    Skips non-dict payloads. Stops after ``[DONE]`` or an ``error`` frame.
    """
    url = f"{api_base.rstrip('/')}/api/v1/query/stream"
    headers: dict[str, str] = {"Accept": "text/event-stream"}
    if turn_id:
        headers["X-Voice-Turn-Id"] = turn_id
    buf = ""
    saw_first_sse = False
    http_t0 = time.perf_counter()
    if timeline:
        timeline.mark("http_stream_open", api_url=url)
    async with client.stream(
        "POST",
        url,
        json=payload,
        headers=headers,
        timeout=120.0,
    ) as r:
        if timeline:
            timeline.mark(
                "http_response_headers",
                status_code=r.status_code,
                ms_http_connect=round((time.perf_counter() - http_t0) * 1000, 2),
            )
        r.raise_for_status()
        async for line in r.aiter_lines():
            if line is None:
                continue
            buf += line + "\n"
            buf, status, objs = _consume_buffer_events(buf)
            for obj in objs:
                first = not saw_first_sse
                if first:
                    saw_first_sse = True
                _log_sse_event(timeline, obj, first_sse=first)
                yield obj
            if status == "done":
                if timeline:
                    timeline.mark("sse_done")
                return
            if status == "error":
                if timeline:
                    timeline.mark("sse_error_frame")
                return
        buf, status, objs = _consume_buffer_events(buf)
        for obj in objs:
            first = not saw_first_sse
            if first:
                saw_first_sse = True
            _log_sse_event(timeline, obj, first_sse=first)
            yield obj
        if timeline and status == "done":
            timeline.mark("sse_done")
