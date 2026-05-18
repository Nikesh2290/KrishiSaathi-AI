"""Vision module tests."""

from __future__ import annotations

import base64
import io
import json

import pytest
from PIL import Image

from modules.vision import engine as vision_engine
from modules.vision.tagger import validate_and_resize_b64_async


def _tiny_jpeg_b64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (10, 120, 30)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.mark.asyncio
async def test_bad_base64():
    with pytest.raises(ValueError):
        await validate_and_resize_b64_async("not!!!")


@pytest.mark.asyncio
async def test_detect_disease_mock(monkeypatch):
    async def fake_gen(*a, **k):
        return json.dumps(
            {
                "is_agricultural": True,
                "image_type": "crop_disease",
                "subject": "wheat",
                "disease": "Yellow Rust",
                "confidence": 0.9,
                "symptoms": ["yellow pustules"],
                "treatment": ["fungicide"],
                "urgency": "high",
            }
        )

    monkeypatch.setattr("modules.vision.engine.generate_with_vision", fake_gen)
    img = _tiny_jpeg_b64()
    out = await vision_engine.detect_disease(img, prefer_local=True)
    assert out["disease"] == "Yellow Rust"
