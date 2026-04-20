import gzip
import json


def test_bundle_builder_returns_expected_shape():
    from offline.bundle_builder import build_bundle

    bundle = build_bundle(state="Punjab", district="Ludhiana")
    assert bundle["state"] == "Punjab"
    assert bundle["district"] == "Ludhiana"
    assert bundle["bundle_version"]
    assert bundle["ttl_hours"] == 24
    data = bundle["data"]
    assert "schemes" in data and isinstance(data["schemes"], list)
    assert "mandi_prices" in data and isinstance(data["mandi_prices"], list)
    assert "crop_calendar" in data and isinstance(data["crop_calendar"], dict)
    assert "weather_history" in data
    for p in data["mandi_prices"]:
        assert p["district"] == "Ludhiana" or p["district"] in {"Karnal"}


def test_bundle_builder_handles_unknown_district_gracefully():
    from offline.bundle_builder import build_bundle

    bundle = build_bundle(state="ZZ", district="Nowhere")
    assert bundle["data"]["mandi_prices"] == []
    assert isinstance(bundle["data"]["schemes"], list)


def test_bundle_version_is_stable_for_same_inputs():
    from offline.bundle_builder import build_bundle

    a = build_bundle(state="Punjab", district="Ludhiana")
    b = build_bundle(state="Punjab", district="Ludhiana")
    assert a["bundle_version"] == b["bundle_version"]


def test_gzip_payload_roundtrip():
    from offline.bundle_builder import build_gzip_bundle

    raw, version = build_gzip_bundle(state="Punjab", district="Ludhiana")
    parsed = json.loads(gzip.decompress(raw))
    assert parsed["bundle_version"] == version
    assert parsed["district"] == "Ludhiana"
