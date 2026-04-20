# KrishiSaathi AI — HTTP API contract

Base URL: `http://localhost:8000` (local). OpenAPI UI: [`/docs`](/docs).

## CORS

Set `CORS_ORIGINS` in `.env` to your frontend origin(s), comma-separated.

---

## `GET /api/v1/health`

**Response 200**

```json
{
  "status": "ok",
  "ollama_ok": true,
  "db_ok": true,
  "version": "0.1.0"
}
```

---

## `POST /api/v1/query`

Main agent call. Returns JSON `AgentResponse`.

**Request body**

```json
{
  "farmer_id": "uuid",
  "query": {
    "text": "मेरी गेहूं की फसल पीली पड़ रही है",
    "voice_b64": null,
    "image_b64": null,
    "language": "hi"
  },
  "context": {
    "location": { "lat": 30.65, "lng": 75.95, "district": "Ludhiana" },
    "connectivity": "online",
    "device_intent": "crop_disease"
  }
}
```

- `connectivity`: `online` | `offline` — drives cached data + routing.
- `device_intent`: e.g. `crop_disease`, `weather`, `scheme_query`, `general`.

**Rate limit:** 10 requests/minute per `farmer_id` (SQLite sliding window).

**Response 200**

```json
{
  "response_id": "uuid",
  "text": "...",
  "structured": {
    "kind": "disease",
    "data": { }
  },
  "data_source": "live",
  "confidence_level": "medium",
  "tool_trace": ["vision", "climate"],
  "language": "hi",
  "timestamp": "2026-04-19T12:00:00Z",
  "safety_flags": []
}
```

**Errors:** `429` rate limit, `422` validation.

---

## `GET /api/v1/farmer/{farmer_id}/twin`

Returns `FarmerTwin` JSON or `404`.

## `PUT /api/v1/farmer/{farmer_id}/twin`

Body: full `FarmerTwin` with matching `farmer_id`.

---

## Language codes

Use ISO-style short codes in `query.language`, e.g. `hi`, `en`, `pa`, `te`.
