"""Farmer digital twin schema."""

from __future__ import annotations

from typing import Any, List, Optional

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
    irrigation: str = "rainfed"


class Financial(BaseModel):
    kcc_loan_amount: float = 0.0
    kcc_bank: str = ""
    pm_fasal_bima: bool = False


class FarmerTwin(BaseModel):
    farmer_id: str
    name: str = ""
    location: Location = Field(default_factory=Location)
    land: Land = Field(default_factory=Land)
    current_crops: List[str] = Field(default_factory=list)
    financial: Financial = Field(default_factory=Financial)
    risk_profile: str = "moderate"
    preferred_language: str = "hi"
    interaction_history: List[Any] = Field(default_factory=list)
