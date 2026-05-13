"""Helpers for integration tests against `POST /api/v1/query/stream`."""

from __future__ import annotations

import json
from typing import Any

import httpx


def _sse_block_to_data_str(block: str) -> str:
    parts: list[str] = []
    for line in block.strip().split("\n"):
        if line.startswith("data:"):
            parts.append(line.removeprefix("data:").strip())
    return "".join(parts).strip()


async def consume_query_stream(
    client: httpx.AsyncClient,
    *,
    path: str,
    json_body: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Returns (parsed_data_objects_in_order, first_error_object_if_any)."""
    out: list[dict[str, Any]] = []
    first_err: dict[str, Any] | None = None
    buf = ""
    async with client.stream(
        "POST",
        path,
        json=json_body,
        headers={"Accept": "text/event-stream"},
        timeout=120.0,
    ) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if line is None:
                continue
            buf += line + "\n"
            while "\n\n" in buf:
                block, buf = buf.split("\n\n", 1)
                data_str = _sse_block_to_data_str(block)
                if not data_str or data_str == "[DONE]":
                    continue
                obj = json.loads(data_str)
                out.append(obj)
                if obj.get("type") == "error" and first_err is None:
                    first_err = obj
    return out, first_err
