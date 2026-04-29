"""Farmer digital twin CRUD."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from config.settings import get_settings
from db.persistence import persist_farmer_twin, resolve_farmer_twin
from models.farmer import FarmerTwin

router = APIRouter(prefix="/api/v1/farmer", tags=["farmer"])


@router.get("/{farmer_id}/twin")
async def get_twin(
    farmer_id: str,
    connectivity: str = Query("online", description="offline = local SQLite only"),
):
    twin = await resolve_farmer_twin(farmer_id, connectivity, get_settings())
    if not twin:
        raise HTTPException(status_code=404, detail="Farmer not found")
    return twin


@router.put("/{farmer_id}/twin")
async def put_twin(
    farmer_id: str,
    body: FarmerTwin,
    connectivity: str = Query("online", description="offline = queue for Supabase sync"),
):
    if body.farmer_id != farmer_id:
        raise HTTPException(status_code=400, detail="farmer_id mismatch")
    await persist_farmer_twin(body, connectivity, get_settings())
    return {"ok": True, "farmer_id": farmer_id}
