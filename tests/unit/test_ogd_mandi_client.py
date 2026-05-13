"""OGD mandi client: parsing and HTTP wiring (mocked)."""

from __future__ import annotations

import httpx
import pytest
import respx

from modules.market import ogd_client


@pytest.mark.asyncio
@respx.mock
async def test_fetch_mandi_prices_normalizes_typical_record() -> None:
    respx.get(ogd_client.OGD_MANDI_RESOURCE).mock(
        return_value=httpx.Response(
            200,
            json={
                "total": 1,
                "records": [
                    {
                        "State": "Punjab",
                        "District": "Ludhiana",
                        "Market": "Khanna",
                        "Commodity": "Wheat",
                        "Variety": "FAQ",
                        "Min Price": "1,950",
                        "Max Price": "2,150",
                        "Modal Price": "2,050",
                        "Price Date": "2026-05-01",
                    }
                ],
            },
        )
    )
    rows = await ogd_client.fetch_mandi_prices(
        "Punjab", "Ludhiana", "Wheat", "test-key", timeout=5.0
    )
    assert len(rows) == 1
    r0 = rows[0]
    assert r0["state"] == "Punjab"
    assert r0["district"] == "Ludhiana"
    assert r0["market"] == "Khanna"
    assert r0["commodity"] == "Wheat"
    assert r0["modal_price"] == 2050.0
    assert r0["min_price"] == 1950.0
    assert r0["max_price"] == 2150.0
    assert r0["price_date"] == "2026-05-01"


@pytest.mark.asyncio
@respx.mock
async def test_fetch_mandi_prices_paginates_until_total() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        q = dict(request.url.params)
        off = int(q.get("offset", "0"))
        if off == 0:
            body = {
                "total": 150,
                "records": [
                    {
                        "State": "Punjab",
                        "District": "Ludhiana",
                        "Market": f"M{i}",
                        "Commodity": "Wheat",
                        "Variety": "",
                        "Modal Price": "2000",
                        "Price Date": "2026-05-01",
                    }
                    for i in range(100)
                ],
            }
        else:
            body = {
                "total": 150,
                "records": [
                    {
                        "State": "Punjab",
                        "District": "Ludhiana",
                        "Market": f"M{i}",
                        "Commodity": "Wheat",
                        "Variety": "",
                        "Modal Price": "2000",
                        "Price Date": "2026-05-01",
                    }
                    for i in range(100, 150)
                ],
            }
        return httpx.Response(200, json=body)

    respx.get(ogd_client.OGD_MANDI_RESOURCE).mock(side_effect=handler)
    rows = await ogd_client.fetch_mandi_prices(
        "Punjab", "Ludhiana", None, "test-key", timeout=10.0
    )
    assert len(rows) == 150


@pytest.mark.asyncio
@respx.mock
async def test_fetch_accepts_district_name_column() -> None:
    respx.get(ogd_client.OGD_MANDI_RESOURCE).mock(
        return_value=httpx.Response(
            200,
            json={
                "total": 1,
                "records": [
                    {
                        "State Name": "Punjab",
                        "District Name": "Ludhiana",
                        "Market Name": "Yard",
                        "Commodity": "Rice",
                        "Modal Price (Rs./Quintal)": "3200",
                        "Price Date": "2026-05-02",
                    }
                ],
            },
        )
    )
    rows = await ogd_client.fetch_mandi_prices(
        "Punjab", "Ludhiana", "Rice", "k", timeout=5.0
    )
    assert len(rows) == 1
    assert rows[0]["district"] == "Ludhiana"
    assert rows[0]["commodity"] == "Rice"
    assert rows[0]["modal_price"] == 3200.0


@pytest.mark.asyncio
@respx.mock
async def test_fetch_retries_with_filters_state_after_keyword_400() -> None:
    def side_effect(request: httpx.Request) -> httpx.Response:
        u = str(request.url)
        if "state.keyword" in u:
            return httpx.Response(400, text="invalid filter field")
        return httpx.Response(
            200,
            json={
                "total": 1,
                "records": [
                    {
                        "State": "Uttar Pradesh",
                        "District": "Gorakhpur",
                        "Market": "Test",
                        "Commodity": "Wheat",
                        "Modal Price": "2100",
                        "Price Date": "2026-05-11",
                    }
                ],
            },
        )

    respx.get(ogd_client.OGD_MANDI_RESOURCE).mock(side_effect=side_effect)
    rows = await ogd_client.fetch_mandi_prices(
        "Uttar Pradesh", "Gorakhpur", None, "test-key", timeout=5.0
    )
    assert len(rows) == 1
    assert rows[0]["district"] == "Gorakhpur"
    assert rows[0]["modal_price"] == 2100.0
