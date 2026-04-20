"""Safety layer tests."""

from __future__ import annotations

from safety.layer import check


def test_unverified_measurement():
    sr = check({}, "Apply 5 mg fungicide per liter")
    assert "[unverified]" in sr.modified_text or "unverified" in sr.modified_text.lower()
