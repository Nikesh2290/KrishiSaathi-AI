"""Sync bundle Pydantic models."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class BundleData(BaseModel):
    schemes: List[Dict[str, Any]] = Field(default_factory=list)
    mandi_prices: List[Dict[str, Any]] = Field(default_factory=list)
    crop_calendar: Dict[str, Any] = Field(default_factory=dict)
    weather_history: List[Dict[str, Any]] = Field(default_factory=list)


class SyncBundle(BaseModel):
    bundle_version: str
    generated_at: str
    district: str
    state: str
    data: BundleData
    ttl_hours: int = 24
