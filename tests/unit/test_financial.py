"""Financial advisor tests."""

from __future__ import annotations

from models.farmer import FarmerTwin
from modules.financial import advisor as fin


def test_advise_sync_numbers():
    t = FarmerTwin(farmer_id="1")
    t.land.total_acres = 4.5
    t.financial.pm_fasal_bima = True
    n = fin.advise_sync(t, season="rabi")
    assert n["kcc_suggested_limit_inr"] == 4.5 * 50000.0
    assert "pmfb_estimated_premium_inr" in n
