"""Utterance language hints for voice, smalltalk, and filler selection."""

from __future__ import annotations

import re
from typing import Literal

LanguageStyle = Literal["hi", "en", "hi-Latn", "mixed"]

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

_HINGLISH_SIGNALS = frozenset({
    "shukriya",
    "dhanyawad",
    "theek",
    "haan",
    "haanji",
    "han",
    "achha",
    "acha",
    "ji",
    "namaste",
    "namaskar",
    "kal",
    "abhi",
    "kk",
    "thx",
    "alvida",
    "milenge",
    "phir",
    "ratri",
    "shubh",
    "hai",
    "hain",
    "meri",
    "mere",
    "kaise",
    "bataiye",
    "fasal",
    "gehu",
    "gahu",
})

_SIMPLE_EN_WORDS = frozenset({
    "the", "is", "are", "was", "what", "when", "where", "which", "who", "how", "why",
    "and", "or", "not", "for", "with", "this", "that", "these", "those", "have", "has",
    "had", "could", "should", "would", "will", "can", "may", "your", "my", "our", "their",
    "today", "tomorrow", "price", "weather", "crop", "disease", "field", "farm", "help",
    "thank", "thanks", "hello", "hi", "yellow", "rust", "wheat", "rice",
})


def detect_language_style(text: str) -> LanguageStyle:
    """Heuristic style for mirroring replies and fillers."""
    t = (text or "").strip()
    if not t:
        return "hi"
    if _DEVANAGARI_RE.search(t):
        # Roman + Devanagari → treat as mixed
        ascii_letters = sum(1 for c in t if ("a" <= c.lower() <= "z"))
        if ascii_letters >= 8:
            return "mixed"
        return "hi"
    lower = t.lower()
    words = set(re.sub(r"[^\w\s]", " ", lower).split())
    words.discard("")
    has_hinglish = bool(words & _HINGLISH_SIGNALS)
    en_score = len(words & _SIMPLE_EN_WORDS)
    if not words:
        return "hi"
    if en_score >= 2 and not has_hinglish:
        return "en"
    if has_hinglish and en_score >= 1:
        return "mixed"
    if has_hinglish:
        return "hi-Latn"
    if en_score >= 1:
        return "mixed"
    return "hi"


def filler_lang_key(style: LanguageStyle, session_locale: str) -> str:
    """Map style to hi/en tables for canned voice fillers."""
    if style == "en":
        return "en"
    if style in ("hi", "hi-Latn", "mixed"):
        base = (session_locale or "hi").split("-", 1)[0].lower()
        return base if base in ("hi", "en") else "hi"
    return "hi"
