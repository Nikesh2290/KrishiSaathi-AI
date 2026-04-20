"""In-memory image ref store with TTL. Survives process lifetime only (hackathon scope)."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class StoredImage:
    data: bytes
    mime: str
    expires_at: int
    farmer_id: str
    purpose: str


_store: Dict[str, StoredImage] = {}


def put(
    data: bytes, mime: str, farmer_id: str, purpose: str, ttl_seconds: int
) -> tuple[str, int]:
    ref = "img_" + secrets.token_hex(12)
    expires_at = int(time.time()) + ttl_seconds
    _store[ref] = StoredImage(
        data=data, mime=mime, farmer_id=farmer_id, purpose=purpose, expires_at=expires_at
    )
    _gc()
    return ref, expires_at


def get(ref: str) -> Optional[StoredImage]:
    _gc()
    return _store.get(ref)


def _gc() -> None:
    now = int(time.time())
    expired = [k for k, v in _store.items() if v.expires_at < now]
    for k in expired:
        _store.pop(k, None)


def clear() -> None:
    _store.clear()
