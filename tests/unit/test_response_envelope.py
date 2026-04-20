from models.response import AgentResponse, StructuredResult


def test_agent_response_new_fields_defaults():
    resp = AgentResponse()
    assert resp.confidence_score == 0.5
    assert resp.model_used == ""
    assert resp.fallback_hint is None


def test_agent_response_accepts_all_fields():
    resp = AgentResponse(
        text="hi",
        confidence_level="high",
        confidence_score=0.88,
        model_used="gemma-4-26b-a4b-it",
        fallback_hint="USE_ONDEVICE",
    )
    d = resp.model_dump(mode="json")
    assert d["confidence_score"] == 0.88
    assert d["model_used"] == "gemma-4-26b-a4b-it"
    assert d["fallback_hint"] == "USE_ONDEVICE"


def test_fallback_hint_rejects_invalid_value():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AgentResponse(fallback_hint="NOT_A_VALID_HINT")  # type: ignore[arg-type]


from response.generator import build


def test_generator_populates_new_fields():
    resp = build(
        draft_text="Test Hindi text.",
        tool_results={"vision_0": {"disease": "Yellow Rust", "confidence": 0.87}},
        tool_trace=["vision"],
        data_source="live",
        language="hi",
        safety_flags=[],
        model_used="gemma-4-26b-a4b-it",
        confidence_score=0.87,
        fallback_hint=None,
    )
    assert resp.model_used == "gemma-4-26b-a4b-it"
    assert resp.confidence_score == 0.87
    assert resp.fallback_hint is None
    assert resp.structured.kind == "disease"
