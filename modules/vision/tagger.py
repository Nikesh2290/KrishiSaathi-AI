"""Validate and resize images before vision API."""

from __future__ import annotations

import base64
import io
from typing import Tuple

from PIL import Image


def validate_and_resize_b64(image_b64: str, max_side: int = 512) -> Tuple[str, str]:
    """Return (normalized_base64, mime_hint). Raises ValueError on bad input."""
    try:
        raw = base64.b64decode(image_b64, validate=True)
    except Exception as e:
        raise ValueError("Invalid base64 image") from e
    try:
        im = Image.open(io.BytesIO(raw))
        im.verify()
    except Exception as e:
        raise ValueError("Not a valid image file") from e
    im = Image.open(io.BytesIO(raw))
    im = im.convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    out = base64.b64encode(buf.getvalue()).decode("ascii")
    return out, "image/jpeg"
