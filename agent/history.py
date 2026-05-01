"""Turn recent query_history rows into planner/synthesis prompt fields."""

from __future__ import annotations

from typing import Dict, List


def build_chat_history_context(turns: List[Dict]) -> List[Dict[str, str]]:
    """Build injected chat history from stored turns (oldest-first).

    ``turns`` items expect keys ``query_text`` and ``response``.
    Assistant responses are whitespace-normalized and truncated to 250 chars for prompts only.
    """
    out: List[Dict[str, str]] = []
    for t in turns:
        u = (t.get("query_text") or "").strip()
        a_raw = " ".join((t.get("response") or "").split())
        a = a_raw[:250] + ("…" if len(a_raw) > 250 else "")
        if u:
            out.append({"u": u, "a": a})
    return out
