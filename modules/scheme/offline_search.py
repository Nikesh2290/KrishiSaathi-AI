"""Keyword / BM25-lite search over scheme_index.json."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List


def _data_path() -> Path:
    return Path(__file__).resolve().parents[2] / "offline" / "data" / "scheme_index.json"


def _tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"\W+", text.lower()) if len(t) > 1]


def keyword_search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    path = _data_path()
    if not path.exists():
        return []
    schemes = json.loads(path.read_text(encoding="utf-8"))
    q_tokens = set(_tokenize(query))
    scores: List[tuple[float, Dict[str, Any]]] = []
    for s in schemes:
        blob = " ".join(
            [
                s.get("name", ""),
                " ".join(s.get("keywords", [])),
                s.get("eligibility", ""),
                s.get("benefits", ""),
            ]
        )
        doc_tokens = _tokenize(blob)
        if not doc_tokens:
            continue
        overlap = len(q_tokens.intersection(set(doc_tokens)))
        score = overlap / math.sqrt(len(doc_tokens) + 1)
        if score > 0:
            scores.append((score, s))
    scores.sort(key=lambda x: -x[0])
    return [s for _, s in scores[:limit]]
