# KrishiSaathi AI — HTTP API Contract (v0.2)

Base URL (local): `http://localhost:8000`. OpenAPI UI: `/docs`.
Backend version: `0.2.0`. Hackathon: Gemma 4 Good.

Most endpoints return JSON. The offline bundle endpoint returns **gzipped JSON**. All errors use the unified envelope (Section 11).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Liveness + Gemma 4 reachability |
| POST | `/api/v1/auth/signup` | Create Supabase user; returns stable `farmer_id` UUID + tokens |
| POST | `/api/v1/auth/login` | Login Supabase user; returns stable `farmer_id` UUID + tokens |
| POST | `/api/v1/query/stream` | Main agent call (Server-Sent Events; canonical) |
| POST | `/api/v1/query/image` | Multipart image upload |
| GET | `/api/v1/sync/bundle` | Offline bundle (district-scoped, gzip) |
| POST | `/api/v1/sync/push` | Push unsynced local SQLite data to Supabase (optional) |
| POST | `/api/v1/market/sync` | Pull OGD mandi prices for `state` + `district` into local SQLite |
| POST | `/api/v1/conversation` | Create a new chat session (`conversation_id` + metadata) |
| GET | `/api/v1/farmer/{farmer_id}/conversations` | List all `conversation_id` values (sessions) for a farmer |
| GET | `/api/v1/farmer/{farmer_id}/conversations/{conversation_id}/history` | Session transcript (`messages` + metadata) |
| DELETE | `/api/v1/farmer/{farmer_id}/conversations/{conversation_id}` | Delete session and all turns (queues for Supabase when offline) |
| GET | `/api/v1/farmer/{farmer_id}/twin` | Read digital twin |
| PUT | `/api/v1/farmer/{farmer_id}/twin` | Update twin |
| POST | `/api/v1/voice/token` | Mint LiveKit room JWT for voice (see Voice section) |

## Voice (LiveKit)

Voice uses **LiveKit** WebRTC: the app obtains a short-lived participant JWT from this API, connects to `server_url` with `participant_token`, and joins `room_name`. A separate **voice worker** process (`python -m voice_agent.worker start`) must be running with `LIVEKIT_*`, `DEEPGRAM_API_KEY` (STT + TTS), optional `DEEPGRAM_TTS_MODEL`, and `KRISHI_API_BASE_URL` set. The worker transcribes speech, calls `POST /api/v1/query/stream`, and plays TTS.

### `POST /api/v1/voice/token`

**Request (JSON)**

- `farmer_id` (string, required): same UUID as auth / query.
- `conversation_id` (string|null): optional session id for query logging continuity.
- `room_name` (string|null): optional; server generates `krishi-{farmer}-{random}` if omitted.
- `participant_identity` (string|null): optional; default `farmer-{first8}`.
- `language` (string): default `hi`; stored in participant metadata for the worker.

**Response 200 (JSON)**

- `server_url` (string): LiveKit URL (usually `wss://...`).
- `room_name` (string): room to join.
- `participant_token` (string): JWT for the client SDK.
- `participant_identity` (string): identity encoded in the JWT (`sub` claim).

**Response 503**

LiveKit env vars are not configured on the API.

Participant JWT **metadata** is JSON: `{"farmer_id","conversation_id","language"}` so the voice worker can call `/query/stream` with the correct farmer context.

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

## 2. `POST /api/v1/auth/signup`

**What it does / used for**

- Creates a Supabase Auth user (email/password) and returns an access token set.
- The returned `farmer_id` is the **stable** user UUID you should use everywhere else (including offline mode).

**Request body (JSON)**

```json
{
  "email": "farmer@example.com",
  "password": "min-6-chars"
}
```

- `email` (string): email address.
- `password` (string): password (min length 6).

**Response 200 (JSON)**

```json
{
  "farmer_id": "uuid",
  "access_token": "jwt",
  "expires_in": 3600,
  "refresh_token": "jwt-or-null"
}
```

- `farmer_id` (string): Supabase Auth user UUID (your primary user id).
- `access_token` (string): bearer token for Supabase APIs.
- `expires_in` (number): seconds until expiry.
- `refresh_token` (string|null): refresh token when available.

## 3. `POST /api/v1/auth/login`

**What it does / used for**

- Logs in an existing Supabase Auth user (email/password) and returns an access token set.

**Request body (JSON)**

```json
{
  "email": "farmer@example.com",
  "password": "your-password"
}
```

- `email` (string): email address.
- `password` (string): password.

**Response 200 (JSON)** — same as signup

```json
{
  "farmer_id": "uuid",
  "access_token": "jwt",
  "expires_in": 3600,
  "refresh_token": "jwt-or-null"
}
```

## 4. `POST /api/v1/conversation`

**Query params**

- `connectivity` (string, optional): default `online`; use `offline` to write local SQLite only until sync.

**Request body (JSON)**

```json
{
  "farmer_id": "uuid-or-demo-id",
  "title": "Optional session title"
}
```

**Response 200**

```json
{
  "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
  "farmer_id": "uuid-or-demo-id",
  "title": "Optional session title",
  "created_at": "2026-05-02T12:00:00.000000+00:00",
  "updated_at": "2026-05-02T12:00:00.000000+00:00"
}
```

## 5. `GET /api/v1/farmer/{farmer_id}/conversations`

**Query params**

- `connectivity` (string, optional): default `online` (prefer Supabase when configured; else SQLite).

**Response 200**

Array of session metadata objects (newest first by `created_at`):

```json
[
  {
    "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
    "farmer_id": "uuid-or-demo-id",
    "title": "Optional session title",
    "created_at": "2026-05-02T12:00:00.000000+00:00",
    "updated_at": "2026-05-02T12:00:00.000000+00:00"
  }
]
```

## 5a. `GET /api/v1/farmer/{farmer_id}/conversations/{conversation_id}/history`

**Query params**

- `connectivity` (string, optional): default `online`; `offline` reads SQLite only.

**Response 200**

Session metadata plus `messages` (oldest first): each message has `id`, `query_text`, `intent`, `response`, `timestamp`, `data_source`, `conversation_id`.

**404** — session missing or `farmer_id` does not own this `conversation_id`.

## 5b. `DELETE /api/v1/farmer/{farmer_id}/conversations/{conversation_id}`

**Query params**

- `connectivity` (string, optional): default `online`. Use `offline` to remove locally and enqueue deletion for the next Supabase sync (`POST /api/v1/sync/push` or app startup sync).

**Response 200**

```json
{
  "deleted": true,
  "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
  "farmer_id": "uuid-or-demo-id"
}
```

**404** — session missing or `farmer_id` does not own this `conversation_id`.

## 6. `POST /api/v1/query/stream` (SSE) — main assistant

Request example (JSON):

```json
{
  "farmer_id": "f1",
  "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
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

**What it does / used for**

- Main assistant call. Streams a natural language answer as `text-delta` events and emits structured metadata at the end.
- Works in online/offline/degraded modes depending on `context.connectivity` and server configuration.

**Request fields**

- `farmer_id` (string, required): stable farmer/user id (Supabase UUID if you use auth; otherwise any stable id for offline-only).
- `conversation_id` (string, optional): thread id from `POST /api/v1/conversation`. When set with `farmer_id`, the server upserts `conversation_metadata` and attaches turns in `query_history` to this session.
- `query` (object, optional): user input payload.
  - `query.text` (string): user text question (default `""`).
  - `query.voice_b64` (string|null): base64 audio (optional; may be ignored).
  - `query.image_ref` (string|null): reference returned by `POST /api/v1/query/image` (Section 7).
  - `query.language` (string): short language code (default `"hi"`).
- `context` (object, optional): device + situation context.
  - `context.location` (object): any location keys (commonly `lat`, `lng`, `district`, `state`).
  - `context.connectivity` (string): `"online"` or `"offline"` (default `"online"`).
  - `context.device_intent` (string): client-side intent hint (default `"general"`).
  - `context.device_capabilities` (object): free-form device capabilities (e.g. `{ "ondevice_model": "..." }`).

**Response 200**

- Content-Type: `text/event-stream`; header `x-vercel-ai-ui-message-stream: v1`.
- Frames are `data: <json>` lines separated by blank lines; stream ends with `data: [DONE]`.
- Assistant text: sum all `{"type":"text-delta","delta":"..."}` payloads (see also `text-start` / `text-end` with shared `id`).
- Final response metadata: `{"type":"data-metadata","data":{...}}` where `data` matches `AgentResponse` **without** the `text` field (use streamed deltas for text).

**Stream error frames**

- `{"type":"error","errorText":"...","errorCode":"IMAGE_REF_EXPIRED","statusCode":404}` — typed `KrishiHTTPException` surfaced on the wire (HTTP status may still be 200).

The non-streaming endpoint `POST /api/v1/query` has been **removed**; clients must use this route.

## 7. `POST /api/v1/query/image`

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

**What it does / used for**

- Uploads an image to the server and returns a temporary `image_ref`.
- Use the `image_ref` in `POST /api/v1/query/stream` as `query.image_ref`.

**Response fields**

- `image_ref` (string): reference token used in subsequent calls.
- `expires_at` (string): ISO UTC timestamp when ref becomes invalid.
- `mime` (string): detected mime type (JPEG/PNG).
- `bytes` (number): original byte size.

## 8. `GET /api/v1/sync/bundle`

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

**What it does / used for**

- Returns a district-scoped offline bundle for on-device/offline mode:
  - schemes index (for RAG)
  - mandi prices (district filtered)
  - crop calendar (global)
  - weather history (district filtered)

## 9. `POST /api/v1/sync/push`

**What it does / used for**

- Pushes unsynced local SQLite data to Supabase (if configured):
  - pending conversation deletions (drained first)
  - farmer twin rows
  - conversation metadata rows
  - query history rows
  - scheme embeddings/vectors

**Request**

- No body.

**Response 200 (JSON)**

If Supabase DB is not configured:

```json
{ "ok": false, "skipped": true, "reason": "Supabase DB not configured" }
```

If sync ran:

```json
{
  "ok": true,
  "conversation_deletes_drained": 0,
  "farmer_twins_synced": 0,
  "conversation_metadata_synced": 0,
  "query_rows_synced": 0,
  "scheme_chunks_synced": 0
}
```

- `conversation_deletes_drained` (number): offline-queued session deletes applied on Supabase.
- `farmer_twins_synced` (number): number of twin rows uploaded.
- `conversation_metadata_synced` (number): number of conversation session rows uploaded.
- `query_rows_synced` (number): number of query logs uploaded.
- `scheme_chunks_synced` (number): number of scheme vector rows uploaded.

## 10. Farmer twin

Unchanged from v0.1:

- `GET /api/v1/farmer/{farmer_id}/twin` → `FarmerTwin` or 404.
- `PUT /api/v1/farmer/{farmer_id}/twin` → body is a full `FarmerTwin`.

### `FarmerTwin` request/response shape (JSON)

```json
{
  "farmer_id": "demo-farmer-1",
  "name": "Ramesh Kumar",
  "location": {
    "state": "Punjab",
    "district": "Ludhiana",
    "village": "Raikot",
    "lat": 30.65,
    "lng": 75.95
  },
  "land": { "total_acres": 4.5, "soil_type": "loamy" },
  "current_crops": ["wheat"],
  "preferred_language": "hi"
}
```

Notes:

- `GET /twin` supports query param `connectivity` (default `online`). Use `offline` to read local SQLite only.
- `PUT /twin` supports query param `connectivity` (default `online`). Use `offline` to queue for later Supabase sync.
- `PUT /twin` requires `body.farmer_id == path farmer_id` (otherwise 400).

## 11. Error envelope

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

## 12. Languages

Short ISO codes in `query.language`: `hi`, `en`, `pa`, `te`, `mr`, `bn`.

## 13. Auth

Optional header `X-Farmer-Id` for future use. Current hackathon build relies on `farmer_id` in the body.
