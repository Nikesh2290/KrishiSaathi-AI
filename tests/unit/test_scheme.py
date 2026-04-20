"""Scheme RAG tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modules.scheme import offline_search


def test_keyword_search_pm_kisan(tmp_path, monkeypatch):
    schemes = [
        {
            "id": "x",
            "name": "PM-KISAN",
            "eligibility": "farmers",
            "benefits": "cash",
            "how_to_apply": "CSC",
            "keywords": ["pm-kisan", "6000"],
        }
    ]
    p = tmp_path / "scheme_index.json"
    p.write_text(json.dumps(schemes), encoding="utf-8")
    monkeypatch.setattr(offline_search, "_data_path", lambda: p)
    hits = offline_search.keyword_search("PM-KISAN eligibility")
    assert len(hits) >= 1
    assert hits[0]["name"] == "PM-KISAN"


@pytest.mark.asyncio
async def test_navigator_mock(monkeypatch):
    from modules.scheme import navigator as nav

    monkeypatch.setattr(
        "modules.scheme.navigator.vector_store.search",
        lambda q, k=5: [{"name": "PM-KISAN", "id": "pm_kisan"}],
    )
    async def fake_gen(*a, **k):
        return "PM-KISAN pays Rs 6000/year."

    monkeypatch.setattr("modules.scheme.navigator.generate", fake_gen)
    from models.farmer import FarmerTwin

    out = await nav.find_schemes(FarmerTwin(farmer_id="1"), "What is PM-KISAN?", True, False)
    assert "answer" in out
