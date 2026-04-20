"""Post-tool safety checks on draft text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


TOXIC = ("kill", "stupid", "hate")

MEDICAL_PATTERN = re.compile(
    r"\b\d+\s*(mg|ml|mcg)\b",
    re.IGNORECASE,
)


@dataclass
class SafetyResult:
    passed: bool
    modified_text: str
    flags: List[str]


def check(
    tool_outputs: Dict[str, Any],
    draft_text: str,
    min_confidence: float = 0.6,
    vision_confidence: float | None = None,
) -> SafetyResult:
    flags: List[str] = []
    text = draft_text

    if MEDICAL_PATTERN.search(text) and "[source]" not in text.lower():
        text = MEDICAL_PATTERN.sub(lambda m: f"{m.group(0)} [unverified]", text)
        flags.append("unsourced_measurement")

    if any(w in text.lower() for w in ("inject", "antibiotic", "veterinary")) and "expert" not in text.lower():
        flags.append("medical_risk")
        text = "For animal health or prescription treatments, please consult a qualified veterinarian or local livestock officer.\n\n" + text

    for w in TOXIC:
        if w in text.lower():
            flags.append("toxic_language")
            text = re.sub(re.escape(w), "[removed]", text, flags=re.IGNORECASE)

    if vision_confidence is not None and vision_confidence < min_confidence:
        flags.append("low_confidence_vision")
        text = "⚠️ Confidence low — please verify with a local agronomist.\n\n" + text

    return SafetyResult(passed=len([f for f in flags if f == "medical_risk"]) == 0, modified_text=text, flags=flags)


def should_escalate(confidence_score: float, threshold: float = 0.70) -> bool:
    """Returns True when the confidence is low enough to warrant re-synthesis on a heavier model."""
    try:
        return float(confidence_score) < float(threshold)
    except (TypeError, ValueError):
        return False
