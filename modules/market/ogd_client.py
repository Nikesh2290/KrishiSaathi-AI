"""HTTP client for data.gov.in OGD mandi daily prices resource."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

OGD_MANDI_RESOURCE = (
    "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
)

# api.data.gov.in is often fronted by a WAF; default ``python-httpx`` UA can see
# timeouts or empty responses while the same URL works in Chrome.
OGD_HTTP_HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 KrishiSaathi/1.0"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-IN,en;q=0.9",
    "Referer": "https://www.data.gov.in/",
}


def format_ogd_http_error(exc: httpx.HTTPStatusError) -> str:
    """Human-readable OGD HTTP failure (no secrets)."""
    r = exc.response
    snippet = (r.text or "")[:800].replace("\n", " ").strip()
    return f"HTTP {r.status_code} from OGD: {snippet or '(empty body)'}"


def _norm_key(key: str) -> str:
    k = str(key).strip().lower()
    k = re.sub(r"[\s()./]+", "_", k)
    k = re.sub(r"_+", "_", k).strip("_")
    return k


def _row_as_norm_map(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {_norm_key(str(k)): v for k, v in raw.items()}


def _pick(norm: Dict[str, Any], *candidates: str) -> Any:
    for c in candidates:
        lc = c.lower()
        for nk, v in norm.items():
            if nk == lc or nk.replace("_", "") == lc.replace("_", ""):
                return v
    return None


def _to_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = re.sub(r"[,\s]", "", str(val).strip())
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _normalize_record(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    n = _row_as_norm_map(raw)
    state = _pick(n, "state", "state_name", "statename")
    district = _pick(n, "district", "district_name", "districtname")
    market = _pick(n, "market", "market_name", "apmc_market", "mandi")
    commodity = _pick(n, "commodity")
    variety = _pick(n, "variety") or ""
    modal = _to_float(
        _pick(
            n,
            "modal_price",
            "modal_pricers_quintal",
            "modal_price_rs_quintal",
        )
    )
    if modal is None:
        modal = _to_float(_pick(n, "modal_price_rs_quintal"))
    if modal is None:
        return None
    min_p = _to_float(
        _pick(
            n,
            "min_price",
            "min_pricers_quintal",
            "min_price_rs_quintal",
        )
    )
    max_p = _to_float(
        _pick(
            n,
            "max_price",
            "max_pricers_quintal",
            "max_price_rs_quintal",
        )
    )
    price_date = _pick(n, "price_date", "arrival_date", "price_dt", "commodity_date")
    if price_date is not None:
        price_date = str(price_date).strip()
    else:
        price_date = ""

    st = str(state or "").strip()
    dist = str(district or "").strip()
    mkt = str(market or "").strip()
    comm = str(commodity or "").strip()
    if not dist or not comm:
        return None
    return {
        "state": st,
        "district": dist,
        "market": mkt or "Unknown",
        "commodity": comm,
        "variety": str(variety or "").strip(),
        "min_price": min_p,
        "max_price": max_p,
        "modal_price": modal,
        "price_date": price_date,
    }


def _parse_ogd_json_response(r: httpx.Response) -> Dict[str, Any]:
    try:
        body = r.json()
    except json.JSONDecodeError as e:
        snippet = (r.text or "")[:500]
        raise ValueError(
            f"OGD returned non-JSON (status {r.status_code}): {snippet!r}"
        ) from e
    if not isinstance(body, dict):
        raise ValueError(f"OGD JSON root is not an object: {type(body).__name__}")
    err = body.get("error") or body.get("message") or body.get("help")
    if err and not body.get("records"):
        raise ValueError(f"OGD error payload: {err!r}")
    return body


async def _download_all_pages(
    client: httpx.AsyncClient,
    api_key: str,
    state: Optional[str],
    district: str,
    commodity: Optional[str],
    state_param_key: Optional[str],
    page_limit: int,
) -> List[Dict[str, Any]]:
    st = (state or "").strip()
    dist = (district or "").strip()
    comm = (commodity or "").strip()

    out: List[Dict[str, Any]] = []
    offset = 0
    while True:
        params: Dict[str, Any] = {
            "api-key": api_key.strip(),
            "format": "json",
            "limit": page_limit,
            "offset": offset,
            "filters[district]": dist,
        }
        if state_param_key and st:
            params[state_param_key] = st
        if comm:
            params["filters[commodity]"] = comm

        r = await client.get(OGD_MANDI_RESOURCE, params=params)
        if r.status_code == 400 and state_param_key == "filters[state.keyword]":
            raise _BadStateFilter()
        if r.is_error:
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise ValueError(format_ogd_http_error(e)) from e

        body = _parse_ogd_json_response(r)
        records = body.get("records") or []
        if not records:
            break

        for raw in records:
            if not isinstance(raw, dict):
                continue
            norm = _normalize_record(raw)
            if norm:
                out.append(norm)

        offset += len(records)
        total = body.get("total")
        try:
            total_i = int(total) if total is not None else offset
        except (TypeError, ValueError):
            total_i = offset
        if offset >= total_i or len(records) < page_limit:
            break

    return out


class _BadStateFilter(Exception):
    """Signal to retry with filters[state] instead of filters[state.keyword]."""


async def fetch_mandi_prices(
    state: Optional[str],
    district: str,
    commodity: Optional[str],
    api_key: str,
    timeout: float = 18.0,
) -> List[Dict[str, Any]]:
    """Fetch mandi records from OGD. Omit ``commodity`` for all commodities in the district.

    Tries ``filters[state.keyword]`` first (OGD docs); on HTTP 400 retries with ``filters[state]``.
    """
    if not api_key or not api_key.strip():
        raise ValueError("OGD API key is required")
    dist = (district or "").strip()
    if not dist:
        raise ValueError("district is required")

    read_s = max(45.0, float(timeout))
    timeout_cfg = httpx.Timeout(read=read_s, connect=min(25.0, read_s), write=read_s, pool=read_s)
    page_limit = 100

    st = (state or "").strip()
    state_key_chain: List[Optional[str]]
    if st:
        state_key_chain = ["filters[state.keyword]", "filters[state]"]
    else:
        state_key_chain = [None]

    for state_param_key in state_key_chain:
        try:
            async with httpx.AsyncClient(
                timeout=timeout_cfg,
                headers=OGD_HTTP_HEADERS,
            ) as client:
                return await _download_all_pages(
                    client,
                    api_key,
                    state,
                    district,
                    commodity,
                    state_param_key,
                    page_limit,
                )
        except _BadStateFilter:
            logger.info(
                "OGD rejected filters[state.keyword] for this resource; retrying with filters[state]"
            )
            continue

    raise RuntimeError("OGD mandi fetch: state filter chain exhausted without result")
