"""Crop/soil vision: Gemma multimodal via gemma_client (classification + analysis)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

from agent.gemma_client import generate_with_vision
from config.settings import Settings, get_settings
from modules.vision.tagger import validate_and_resize_b64_async

logger = logging.getLogger(__name__)

SYSTEM_ANALYZE = """You are KrishiSaathi image analyst for Indian farmers.
Look at the image and decide if it is agricultural domain.

AGRICULTURAL means: crop plants, leaves, fruits on plant, field/soil close-ups,
farm pests on crops, irrigation/farm equipment ONLY when crop/soil is clearly visible.

NOT agricultural: random people, selfies, animals unrelated to farm work, vehicles,
buildings, indoor scenes without crops, food dishes, screenshots, documents, etc.

Respond with ONLY valid JSON (no markdown).

If NOT agricultural:
{"is_agricultural": false, "image_type": "person"|"animal"|"vehicle"|"building"|"food"|"document"|"other",
 "description": "<one short sentence describing what you see>",
 "specialization_note": "I specialize in crop disease, soil health, schemes, weather, and farm advice. Upload a clear photo of your crop or soil for accurate help.",
 "confidence": <float 0-1>}

If agricultural:
{"is_agricultural": true, "image_type": "crop_disease"|"soil"|"crop_healthy"|"crop_general",
 "subject": "<crop/plant name or soil>",
 "disease": "<disease name OR abiotic stress OR \"Healthy\" OR \"Unclear\">",
 "confidence": <float 0-1>,
 "symptoms": ["<visible signs if any>"],
 "treatment": ["<practical steps; empty if only describing soil/healthy plant>"],
 "urgency": "low"|"medium"|"high"}

Rules:
- Answer the farmer's question in the agricultural branch via disease/symptoms/treatment fields.
- If unsure on disease, set confidence below 0.5 and disease \"Unknown\".
- description and specialization_note must be omitted when is_agricultural is true."""

DEFAULT_SPEC_NOTE = (
    "I specialize in crop disease, soil health, schemes, weather, and farm advice. "
    "Upload a clear photo of your crop or soil for accurate help."
)


def _strip_json_fence(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n", "", raw)
        raw = re.sub(r"\n```$", "", raw)
    return raw


def _parse_vision_json(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Vision JSON parse failed: %s", raw[:200])
        return {
            "is_agricultural": True,
            "image_type": "crop_general",
            "subject": "unknown",
            "disease": "Unknown",
            "confidence": 0.4,
            "symptoms": [],
            "treatment": ["Consult a local agronomist with the photo."],
            "urgency": "medium",
            "parse_error": True,
        }


async def analyze_image(
    image_b64: str | None,
    user_query: str,
    prefer_local: bool,
    settings: Settings | None = None,
    *,
    source_mime: str | None = None,
) -> Dict[str, Any]:
    settings = settings or get_settings()
    q = (user_query or "").strip()
    if not image_b64:
        return {
            "is_agricultural": True,
            "image_type": "crop_general",
            "subject": "",
            "disease": "No image",
            "confidence": 0.0,
            "symptoms": [],
            "treatment": ["Please upload a clear photo of the affected crop or soil."],
            "urgency": "low",
            "note": "no_image",
        }

    norm_b64, norm_mime = await validate_and_resize_b64_async(image_b64)
    vision_mime = norm_mime or source_mime or "image/jpeg"
    user = (
        f'Farmer question (answer via JSON fields; same language tone as question): "{q}"\n'
        "Return ONLY one JSON object as specified in your instructions."
    )
    raw = await generate_with_vision(
        SYSTEM_ANALYZE,
        user,
        norm_b64,
        prefer_local=prefer_local,
        settings=settings,
        image_mime=vision_mime,
    )
    data = _parse_vision_json(_strip_json_fence(raw))

    is_ag = data.get("is_agricultural")
    if is_ag is False:
        out = {
            "is_agricultural": False,
            "image_type": str(data.get("image_type") or "other"),
            "description": str(data.get("description") or "").strip()
            or "Could not describe the image in detail.",
            "specialization_note": str(data.get("specialization_note") or "").strip()
            or DEFAULT_SPEC_NOTE,
            "confidence": float(data.get("confidence") or 0.9),
        }
        return out

    # Agricultural branch — normalize required keys
    conf = float(data.get("confidence") or 0.0)
    out_ag: Dict[str, Any] = {
        "is_agricultural": True,
        "image_type": str(data.get("image_type") or "crop_general"),
        "subject": str(data.get("subject") or "").strip() or "crop",
        "disease": data.get("disease"),
        "confidence": conf,
        "symptoms": list(data.get("symptoms") or []),
        "treatment": list(data.get("treatment") or []),
        "urgency": str(data.get("urgency") or "medium"),
    }
    if out_ag["disease"] is None:
        out_ag["disease"] = "Unknown"
    else:
        out_ag["disease"] = str(out_ag["disease"])

    if conf < 0.5:
        out_ag["treatment"] = list(out_ag["treatment"]) + [
            "[unverified] Low confidence — verify with field expert."
        ]
    return out_ag


from modules.vision import image_store as _image_store


async def detect_disease(
    image_b64: str | None,
    prefer_local: bool,
    settings: Settings | None = None,
    user_query: str = "",
) -> Dict[str, Any]:
    return await analyze_image(image_b64, user_query, prefer_local, settings)


async def detect_disease_bytes(
    data: bytes,
    mime: str,
    prefer_local,
    settings,
    user_query: str = "",
):
    import base64

    b64 = base64.b64encode(data).decode("ascii")
    return await analyze_image(
        b64, user_query, prefer_local, settings, source_mime=mime
    )


async def detect_disease_by_ref(
    image_ref,
    prefer_local,
    settings,
    user_query: str = "",
):
    if not image_ref:
        return {
            "is_agricultural": True,
            "disease": None,
            "confidence": 0.0,
            "symptoms": [],
            "treatment": ["Please upload a clear photo of the affected crop or soil."],
            "urgency": "low",
            "note": "no_image",
        }
    stored = _image_store.get(image_ref)
    if stored is None:
        from models.errors import ErrorCode, KrishiHTTPException

        raise KrishiHTTPException(
            status_code=404,
            code=ErrorCode.IMAGE_REF_EXPIRED,
            message="image_ref is unknown or expired",
        )
    return await detect_disease_bytes(
        stored.data, stored.mime, prefer_local, settings, user_query=user_query
    )
