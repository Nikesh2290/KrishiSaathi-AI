from agent.language import detect_language_style, filler_lang_key


def test_detect_english():
    assert detect_language_style("What is the mandi price for wheat today?") == "en"


def test_detect_hinglish_mix():
    s = detect_language_style("mere wheat mein yellow rust aa gaya")
    assert s in ("mixed", "hi-Latn")


def test_filler_lang_key():
    assert filler_lang_key("en", "hi") == "en"
    assert filler_lang_key("mixed", "hi") == "hi"
