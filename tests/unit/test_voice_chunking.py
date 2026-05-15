"""Voice worker sentence chunking helpers."""

from __future__ import annotations

from voice_agent import worker as w


def test_sentence_boundary_danda():
    buf = "नमस्ते।दूसरा"
    cut = w._sentence_boundary(buf)
    assert cut == len("नमस्ते।")
    assert buf[:cut].strip() == "नमस्ते।"


def test_sentence_boundary_decimal_skips_dot():
    buf = "Use 1.5 kg per acre. Next sentence."
    cut = w._sentence_boundary(buf)
    # first . is between digits — skip; next boundary is after second .
    assert cut is not None
    assert buf[:cut].rstrip().endswith(".")


def test_min_flush_strong_boundary():
    assert w._is_strong_flush_boundary("ठीक है।")
    assert w._is_strong_flush_boundary("OK!")
    assert not w._is_strong_flush_boundary("Hi.")


def test_strip_overlap_prefix():
    prev = "prefix tailend"
    chunk = "tailend and the rest"
    out = w._strip_chunk_overlap(chunk, prev, min_ov=4, max_ov=20)
    assert out == "and the rest"


def test_strip_overlap_no_false_positive():
    chunk = "completely new"
    assert w._strip_chunk_overlap(chunk, "other text") == chunk


def test_build_voice_welcome_tier1():
    from models.farmer import FarmerTwin, Land, Location

    twin = FarmerTwin(
        farmer_id="f1",
        name="Ramesh",
        location=Location(state="Punjab", district="Ludhiana"),
        land=Land(total_acres=4.5),
        current_crops=["wheat", "mustard"],
    )
    msg = w._build_voice_welcome(twin, "hi")
    assert "Ramesh" in msg
    assert "Ludhiana" in msg
    assert "गेहूं" in msg or "wheat" in msg.lower()


def test_build_voice_welcome_empty():
    msg = w._build_voice_welcome(None, "hi")
    assert "Krishi Saathi" in msg


def test_format_filler_without_name():
    s = w._format_filler("{name} ji, hello", None)
    assert "{name}" not in s
    assert "hello" in s
