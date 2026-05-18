"""Validate and resize images before vision API."""

from __future__ import annotations

import asyncio
import base64
import io
from typing import Tuple

from PIL import Image


def _resize_sync(image_b64: str, max_side: int) -> Tuple[str, str]:
    raw = base64.b64decode(image_b64, validate=True)
    im = Image.open(io.BytesIO(raw))
    im.verify()
    im = Image.open(io.BytesIO(raw))
    im = im.convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    out = base64.b64encode(buf.getvalue()).decode("ascii")
    return out, "image/jpeg"


async def validate_and_resize_b64_async(image_b64: str, max_side: int = 512) -> Tuple[str, str]:
    """Return (normalized_base64, mime_hint). Runs PIL ops in thread pool."""
    try:
        return await asyncio.to_thread(_resize_sync, image_b64, max_side)
    except Exception as e:
        raise ValueError(f"Image processing failed: {e}") from e
