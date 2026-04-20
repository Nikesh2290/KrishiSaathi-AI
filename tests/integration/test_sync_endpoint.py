import gzip
import json

import pytest
from httpx import ASGITransport, AsyncClient


async def _fetch_raw(client: AsyncClient, params: dict) -> tuple[int, dict, bytes]:
    """Fetch the sync bundle and return status, headers, and raw (undecoded) body.

    httpx automatically decodes responses with ``Content-Encoding: gzip``.
    To validate the server-side gzip compression we bypass decoding by
    streaming the raw bytes.
    """
    async with client.stream("GET", "/api/v1/sync/bundle", params=params) as response:
        raw = b""
        async for chunk in response.aiter_raw():
            raw += chunk
        return response.status_code, dict(response.headers), raw


@pytest.mark.asyncio
async def test_sync_bundle_returns_gzipped_json():
    from api.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t"
    ) as client:
        status, headers, raw = await _fetch_raw(
            client, {"state": "Punjab", "district": "Ludhiana"}
        )
    assert status == 200, raw
    assert headers.get("content-encoding") == "gzip"
    payload = json.loads(gzip.decompress(raw))
    assert payload["district"] == "Ludhiana"
    assert payload["state"] == "Punjab"
    assert payload["bundle_version"]
    assert "schemes" in payload["data"]


@pytest.mark.asyncio
async def test_sync_bundle_missing_params_returns_400_envelope():
    from api.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t"
    ) as client:
        r = await client.get("/api/v1/sync/bundle", params={"state": "Punjab"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_sync_bundle_returns_304_for_matching_version():
    from api.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t"
    ) as client:
        status1, _, raw1 = await _fetch_raw(
            client, {"state": "Punjab", "district": "Ludhiana"}
        )
        assert status1 == 200
        version = json.loads(gzip.decompress(raw1))["bundle_version"]

        r2 = await client.get(
            "/api/v1/sync/bundle",
            params={
                "state": "Punjab",
                "district": "Ludhiana",
                "bundle_version": version,
            },
        )
    assert r2.status_code == 304
