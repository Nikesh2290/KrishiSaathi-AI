"""Farmer digital twin schema."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Location(BaseModel):
    state: str = ""
    district: str = ""
    village: str = ""
    lat: Optional[float] = None
    lng: Optional[float] = None


class Land(BaseModel):
    total_acres: float = 0.0
    soil_type: str = "loamy"


class FarmerTwin(BaseModel):
    farmer_id: str
    name: str = ""
    location: Location = Field(default_factory=Location)
    land: Land = Field(default_factory=Land)
    current_crops: List[str] = Field(default_factory=list)
    preferred_language: str = "en"
