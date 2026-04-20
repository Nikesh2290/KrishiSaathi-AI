"""Farmer twin CRUD tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app
from models.farmer import FarmerTwin


def test_farmer_put_get():
    with TestClient(app) as client:
        twin = FarmerTwin(
            farmer_id="t1",
            name="Ramesh",
        )
        r = client.put("/api/v1/farmer/t1/twin", json=twin.model_dump())
        assert r.status_code == 200
        g = client.get("/api/v1/farmer/t1/twin")
        assert g.status_code == 200
        assert g.json()["name"] == "Ramesh"
