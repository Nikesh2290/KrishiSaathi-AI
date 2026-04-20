"""Crop disease: Gemma multimodal via gemma_client."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

from agent.gemma_client import generate_with_vision
from config.settings import Settings, get_settings
from modules.vision.tagger import validate_and_resize_b64

logger = logging.getLogger(__name__)

SYSTEM = """You are an agricultural assistant. Analyze the crop photo for visible disease or stress.
Respond with ONLY valid JSON:
{"disease": str, "confidence": float 0-1, "symptoms": [str], "treatment": [str], "urgency": "low"|"medium"|"high"}
If unsure, set confidence below 0.5 and say disease: "Unknown"."""


async def detect_disease(
    image_b64: str | None,
    prefer_local: bool,
    settings: Settings | None = None,
) -> Dict[str, Any]:
    settings = settings or get_settings()
    if not image_b64:
        return {
            "disease": "No image",
            "confidence": 0.0,
            "symptoms": [],
            "treatment": ["Please upload a clear photo of the affected crop."],
            "urgency": "low",
            "note": "no_image",
        }
    norm_b64, _ = validate_and_resize_b64(image_b64)
    user = "Identify disease or abiotic stress. JSON only."
    raw = await generate_with_vision(SYSTEM, user, norm_b64, prefer_local=prefer_local, settings=settings)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n", "", raw)
        raw = re.sub(r"\n```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Vision JSON parse failed: %s", raw[:200])
        data = {
            "disease": "Unknown",
            "confidence": 0.4,
            "symptoms": [],
            "treatment": ["Consult a local agronomist with the photo."],
            "urgency": "medium",
        }
    conf = float(data.get("confidence") or 0.0)
    if conf < 0.5:
        data["treatment"] = list(data.get("treatment") or []) + ["[unverified] Low confidence — verify with field expert."]
    return data


from modules.vision import image_store as _image_store


async def detect_disease_bytes(data: bytes, mime: str, prefer_local, settings):
    import base64

    b64 = base64.b64encode(data).decode("ascii")
    return await detect_disease(b64, prefer_local, settings)


async def detect_disease_by_ref(image_ref, prefer_local, settings):
    if not image_ref:
        return {"disease": None, "confidence": 0.0, "note": "no_image"}
    stored = _image_store.get(image_ref)
    if stored is None:
        from models.errors import ErrorCode, KrishiHTTPException

        raise KrishiHTTPException(
            status_code=404,
            code=ErrorCode.IMAGE_REF_EXPIRED,
            message="image_ref is unknown or expired",
        )
    return await detect_disease_bytes(stored.data, stored.mime, prefer_local, settings)
