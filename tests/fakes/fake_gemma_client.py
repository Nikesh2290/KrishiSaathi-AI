"""Deterministic fake for agent.gemma_client used in all non-live tests."""

from __future__ import annotations

import json
from typing import Any, List, Optional, Sequence, Union


class FakeGemmaClient:
    def __init__(self) -> None:
        self.calls: List[dict] = []
        self.plan_response: dict = {
            "tools": [{"tool": "scheme", "params": {"query": "PM-KISAN"}}]
        }
        self.synth_text: str = "Farmer-friendly summary of tool results."
        self.heavy_synth_text: str = "Refined answer from 31b."

    async def generate(
        self,
        messages: Sequence[Union[dict, Any]],
        prefer_local: bool = False,
        settings: Any = None,
        heavy: bool = False,
        voice_mode: bool = False,
    ) -> str:
        self.calls.append({"messages": list(messages), "heavy": heavy})
        first_system = next(
            (m for m in messages if (isinstance(m, dict) and m.get("role") == "system")),
            None,
        )
        sys_content = (first_system.get("content") if first_system else "") or ""
        sys_lower = sys_content.lower()
        if first_system and (
            "planning component" in sys_lower or "intent classifier" in sys_lower
        ):
            return json.dumps(self.plan_response)
        return self.heavy_synth_text if heavy else self.synth_text

    async def generate_with_vision(
        self,
        system_prompt: str,
        user_text: str,
        image_bytes: bytes,
        prefer_local: bool = False,
        settings: Any = None,
        *,
        image_mime: str = "image/jpeg",
        voice_mode: bool = False,
    ) -> str:
        self.calls.append({"vision": True, "bytes": len(image_bytes), "image_mime": image_mime})
        return json.dumps(
            {"disease": "Yellow Rust", "confidence": 0.87, "treatment": ["Propiconazole"]}
        )


def install_fake(monkeypatch, fake: Optional[FakeGemmaClient] = None) -> FakeGemmaClient:
    fake = fake or FakeGemmaClient()
    import agent.gemma_client as real

    monkeypatch.setattr(real, "generate", fake.generate)
    monkeypatch.setattr(real, "generate_with_vision", fake.generate_with_vision)
    return fake
