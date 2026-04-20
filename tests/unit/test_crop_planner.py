"""Crop planner tests."""

from __future__ import annotations

from modules.crop_planner import planner as cp


def test_recommend_deterministic():
    out = cp.recommend("loamy", "Punjab", "rabi", "tube_well", prefer_local=True)
    assert "top_crops" in out
    assert len(out["top_crops"]) >= 1
