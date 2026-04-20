# KrishiSaathi AI — HTTP API Contract (v0.2)

Base URL (local): `http://localhost:8000`. OpenAPI UI: `/docs`.
Backend version: `0.2.0`. Hackathon: Gemma 4 Good.

All endpoints return JSON. All errors use the unified envelope (Section 6).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Liveness + Gemma 4 reachability |
| POST | `/api/v1/query` | Main agent call |
| POST | `/api/v1/query/image` | Multipart image upload |
| GET | `/api/v1/sync/bundle` | Offline bundle (district-scoped, gzip) |
| GET | `/api/v1/farmer/{farmer_id}/twin` | Read digital twin |
| PUT | `/api/v1/farmer/{farmer_id}/twin` | Update twin |

## 1. `GET /api/v1/health`

```json
{
  "status": "ok",
  "version": "0.2.0",
  "ai_studio_ok": true,
  "ollama_ok": false,
  "db_ok": true,
  "chroma_ok": true,
  "gemma4_model_configured": "gemma-4-26b-a4b-it"
}
```

## 2. `POST /api/v1/query`

Request:

```json
{
  "farmer_id": "f1",
  "query": {
    "text": "मेरी गेहूं की फसल पीली पड़ रही है",
    "image_ref": null,
    "language": "hi"
  },
  "context": {
    "location": { "lat": 30.65, "lng": 75.95, "district": "Ludhiana", "state": "Punjab" },
    "connectivity": "online",
    "device_intent": "crop_disease",
    "device_capabilities": { "ondevice_model": "gemma-4-e4b-it" }
  }
}
```

- `query.image_ref` — returned by `POST /query/image` (Section 3).
- `context.connectivity` — `online | offline | degraded`.
- `context.device_intent` — `crop_disease | scheme_query | market_price | financial | weather | crop_plan | general | alert`.

Response 200:

```json
{
  "response_id": "uuid",
  "text": "...",
  "structured": { "kind": "disease", "data": {} },
  "data_source": "live",
  "confidence_level": "high",
  "confidence_score": 0.87,
  "model_used": "gemma-4-26b-a4b-it",
  "tool_trace": ["vision", "climate"],
  "safety_flags": [],
  "fallback_hint": null,
  "language": "hi",
  "timestamp": "2026-04-20T12:00:00Z"
}
```

## 3. `POST /api/v1/query/image`

`multipart/form-data`:

| Field | Type | Required |
|---|---|---|
| `image` | file (JPEG/PNG, ≤ 5 MB) | yes |
| `farmer_id` | string | yes |
| `purpose` | string (`crop_disease` \| `soil_photo` \| `pest_id`) | yes |

Response 201:

```json
{
  "image_ref": "img_7a3f...",
  "expires_at": "2026-04-20T13:00:00Z",
  "mime": "image/jpeg",
  "bytes": 482113
}
```

## 4. `GET /api/v1/sync/bundle`

Query params: `state` (required), `district` (required), `bundle_version` (optional).

- `200 OK` — gzipped JSON body; headers: `Content-Encoding: gzip`, `X-Bundle-Version: <id>`.
- `304 Not Modified` — when `bundle_version` matches current server bundle.

Payload (after gunzip):

```json
{
  "bundle_version": "punjab-ludhiana-abc123",
  "generated_at": "2026-04-20T06:00:00Z",
  "district": "Ludhiana",
  "state": "Punjab",
  "data": {
    "schemes": [],
    "mandi_prices": [],
    "crop_calendar": {},
    "weather_history": []
  },
  "ttl_hours": 24
}
```

## 5. Farmer twin

Unchanged from v0.1:

- `GET /api/v1/farmer/{farmer_id}/twin` → `FarmerTwin` or 404.
- `PUT /api/v1/farmer/{farmer_id}/twin` → body is a full `FarmerTwin`.

## 6. Error envelope

Every non-2xx response:

```json
{
  "error": {
    "code": "UPSTREAM_RATE_LIMIT",
    "message": "quota exhausted",
    "retryable": true,
    "retry_after_seconds": 30,
    "fallback_hint": "USE_ONDEVICE"
  }
}
```

| HTTP | `error.code` | `retryable` | `fallback_hint` |
|---|---|---|---|
| 400 | `VALIDATION_ERROR` | false | `null` |
| 404 | `FARMER_NOT_FOUND` | false | `null` |
| 404 | `IMAGE_REF_EXPIRED` | false | `null` |
| 408 | `LLM_TIMEOUT` | true | `USE_ONDEVICE` |
| 413 | `IMAGE_TOO_LARGE` | false | `null` |
| 415 | `IMAGE_UNSUPPORTED_TYPE` | false | `null` |
| 429 | `UPSTREAM_RATE_LIMIT` | true | `USE_ONDEVICE` |
| 503 | `UPSTREAM_UNAVAILABLE` | true | `USE_ONDEVICE` |
| 500 | `INTERNAL_ERROR` | false | `RETRY_ONLINE_LATER` |

## 7. Languages

Short ISO codes in `query.language`: `hi`, `en`, `pa`, `te`, `mr`, `bn`.

## 8. Auth

Optional header `X-Farmer-Id` for future use. Current hackathon build relies on `farmer_id` in the body.
