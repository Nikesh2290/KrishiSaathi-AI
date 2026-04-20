import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_post_query_image_returns_ref():
    from api.main import app

    with open("tests/fixtures/wheat_rust.jpg", "rb") as f:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://t"
        ) as client:
            r = await client.post(
                "/api/v1/query/image",
                files={"image": ("wheat.jpg", f.read(), "image/jpeg")},
                data={"farmer_id": "f1", "purpose": "crop_disease"},
            )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["image_ref"].startswith("img_")
    assert body["mime"] == "image/jpeg"
    assert body["bytes"] > 0


@pytest.mark.asyncio
async def test_post_query_image_rejects_large_file():
    from api.main import app

    big = b"\xff\xd8\xff" + b"A" * (6 * 1024 * 1024)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t"
    ) as client:
        r = await client.post(
            "/api/v1/query/image",
            files={"image": ("big.jpg", big, "image/jpeg")},
            data={"farmer_id": "f1", "purpose": "crop_disease"},
        )
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "IMAGE_TOO_LARGE"


@pytest.mark.asyncio
async def test_post_query_image_rejects_bad_mime():
    from api.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t"
    ) as client:
        r = await client.post(
            "/api/v1/query/image",
            files={"image": ("data.bin", b"hello", "application/octet-stream")},
            data={"farmer_id": "f1", "purpose": "crop_disease"},
        )
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "IMAGE_UNSUPPORTED_TYPE"


@pytest.mark.asyncio
async def test_image_ref_flows_into_query(monkeypatch):
    from tests.fakes.fake_gemma_client import install_fake
    from api.main import app

    install_fake(monkeypatch)

    with open("tests/fixtures/wheat_rust.jpg", "rb") as f:
        img_bytes = f.read()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t"
    ) as client:
        r = await client.post(
            "/api/v1/query/image",
            files={"image": ("wheat.jpg", img_bytes, "image/jpeg")},
            data={"farmer_id": "f1", "purpose": "crop_disease"},
        )
        assert r.status_code == 201, r.text
        image_ref = r.json()["image_ref"]

        r = await client.post(
            "/api/v1/query",
            json={
                "farmer_id": "f1",
                "query": {
                    "text": "मेरी गेहूं की फसल पीली पड़ रही है",
                    "image_ref": image_ref,
                    "language": "hi",
                },
                "context": {
                    "connectivity": "online",
                    "device_intent": "crop_disease",
                    "location": {"district": "Ludhiana", "state": "Punjab"},
                },
            },
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_used"].startswith("gemma-4-")
    assert "vision" in body["tool_trace"]


@pytest.mark.asyncio
async def test_expired_image_ref_returns_404_envelope(monkeypatch):
    from tests.fakes.fake_gemma_client import install_fake
    from api.main import app

    install_fake(monkeypatch)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t"
    ) as client:
        r = await client.post(
            "/api/v1/query",
            json={
                "farmer_id": "f1",
                "query": {"text": "photo?", "image_ref": "img_deadbeef"},
                "context": {"connectivity": "online", "device_intent": "crop_disease"},
            },
        )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "IMAGE_REF_EXPIRED"
