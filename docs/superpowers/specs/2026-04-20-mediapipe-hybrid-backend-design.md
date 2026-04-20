# KrishiSaathi AI — MediaPipe Hybrid Backend Design

**Date:** 2026-04-20
**Hackathon:** Gemma 4 Good Hackathon (Kaggle × Google DeepMind)
**Submission deadline:** 2026-05-18 23:59 UTC
**Status:** Draft — awaiting user review
**Scope:** Backend-only changes in this repo. React Native frontend lives in a separate repo.

---

## 1. Goal

Adapt this FastAPI backend to serve a React Native companion app that uses **MediaPipe LLM Inference** with on-device Gemma 4 E4B/E2B for offline work, while keeping **Gemma 4 26B A4B / 31B** on Google AI Studio for online heavy reasoning. The result is a hackathon submission that:

1. Satisfies the rule "use at least one Gemma 4 model" with explicit Gemma 4 variant ids.
2. Has a credible offline-first story judges can verify on camera (airplane-mode demo).
3. Gives the RN team an unambiguous, stable contract (endpoints, response shapes, routing rules, fallback semantics).
4. Ships in ~4.5 focused dev days with a working public deploy before May 18.

## 2. Constraints

- Free-tier only (no paid Vertex AI, no paid Gemini, no paid hosting).
- Backend deployable on Render or Fly.io free tier.
- Python 3.11+, FastAPI, LangGraph, ChromaDB, SQLite; no new heavy dependencies.
- Must work when the user's phone is offline (RN app owns that path; backend's job is to ship the right bundle beforehand).
- Judges will run `docker compose up --build` blind on a clean clone — README must be correct.

## 3. Model map (Gemma 4 only)

| Layer | Variant | Runtime |
|---|---|---|
| Backend planner & synthesizer (primary) | `gemma-4-26b-a4b-it` | Google AI Studio free tier |
| Backend heavy reasoning (escalation) | `gemma-4-31b-it` | Google AI Studio free tier |
| On-device primary (RN, MediaPipe) | `gemma-4-e4b-it` | MediaPipe LLM Inference on Android |
| On-device low-RAM fallback | `gemma-4-e2b-it` | MediaPipe LLM Inference on Android |

All four are Gemma 4 (released 2026-04-02, Apache 2.0). Gemma 3n and Gemma 2 references are removed everywhere in this repo.

## 4. Routing policy (intent-based)

The RN app decides where each query runs; the backend's behavior is stateless with respect to that choice.

| Connectivity | Intent | Path |
|---|---|---|
| online | `crop_disease`, `scheme_query`, `market_price`, `financial` | Backend `/query` (Gemma 4 26B A4B; escalates to 31B if conf < 0.70) |
| online | `weather`, `crop_plan`, `general`, `alert` | On-device Gemma 4 E4B/E2B |
| offline | any | On-device Gemma 4 E4B/E2B |

**Low-confidence offline behavior (RN app responsibility):** If on-device returns `confidence < 0.70`, show the answer with a non-blocking CTA ("Get expert analysis — needs internet"). If the user taps and is online, the RN app calls `POST /query` with the same inputs.

**Upstream 429 degradation (backend + RN responsibility):** Backend returns `fallback_hint: "USE_ONDEVICE"` in the error envelope; RN app silently re-runs the query on-device and shows a "network busy" banner.

## 5. API contract (v0.2)

### 5.1 Endpoint inventory

| Method | Path | Status |
|---|---|---|
| `GET` | `/api/v1/health` | unchanged (new fields) |
| `POST` | `/api/v1/query` | modified |
| `POST` | `/api/v1/query/image` | new |
| `GET` | `/api/v1/sync/bundle` | new |
| `GET` | `/api/v1/farmer/{farmer_id}/twin` | unchanged |
| `PUT` | `/api/v1/farmer/{farmer_id}/twin` | unchanged |
| ~~`GET` `/api/v1/query/stream`~~ | | removed |

### 5.2 `POST /api/v1/query`

**Request**

```json
{
  "farmer_id": "f1",
  "query": {
    "text": "मेरी गेहूं की फसल पीली पड़ रही है",
    "language": "hi",
    "image_ref": null
  },
  "context": {
    "location": { "lat": 30.65, "lng": 75.95, "district": "Ludhiana", "state": "Punjab" },
    "connectivity": "online",
    "device_intent": "crop_disease",
    "device_capabilities": { "ondevice_model": "gemma-4-e4b-it" }
  }
}
```

- `query.image_b64` is removed. Use `POST /query/image` first, then pass `query.image_ref`.
- `context.connectivity`: `"online"` | `"offline"` | `"degraded"`.
- `context.device_capabilities.ondevice_model` is optional; informs logging and `fallback_hint` decisions.

**Response 200**

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

New fields: `confidence_score` (float 0–1), `model_used` (Gemma 4 variant id string), `fallback_hint` (`null` | `"USE_ONDEVICE"` | `"RETRY_ONLINE_LATER"`).

### 5.3 Error envelope (all errors)

```json
{
  "error": {
    "code": "UPSTREAM_RATE_LIMIT",
    "message": "AI Studio quota exhausted for this minute.",
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

### 5.4 `POST /api/v1/query/image`

**Request:** `multipart/form-data` with fields `image` (file, JPEG/PNG, max 5 MB), `farmer_id`, `purpose` (`crop_disease` | `soil_photo` | `pest_id`).

**Response 201**

```json
{
  "image_ref": "img_7a3f9c...",
  "expires_at": "2026-04-20T13:00:00Z",
  "mime": "image/jpeg",
  "bytes": 482113
}
```

`image_ref` TTL is 1 hour; expired refs return 404 when used in `/query`.

### 5.5 `GET /api/v1/sync/bundle`

**Query params:** `state` (required), `district` (required), `bundle_version` (optional).

**Response 200** (gzip JSON, ~1–2 MB):

```json
{
  "bundle_version": "2026-04-20-punjab-ludhiana-v3",
  "generated_at": "2026-04-20T06:00:00Z",
  "district": "Ludhiana",
  "state": "Punjab",
  "data": {
    "schemes": [],
    "mandi_prices": [],
    "crop_calendar": {},
    "weather_history": {}
  },
  "ttl_hours": 24
}
```

If `bundle_version` matches server state, return `304 Not Modified`. Server caches built bundles in memory for 1 hour.

### 5.6 `GET /api/v1/health`

```json
{
  "status": "ok",
  "version": "0.2.0",
  "ai_studio_ok": true,
  "ollama_ok": true,
  "db_ok": true,
  "chroma_ok": true,
  "gemma4_model_configured": "gemma-4-26b-a4b-it"
}
```

## 6. File structure

### 6.1 Deletions

- `api/middleware/rate_limit.py`
- `agent/orchestrator.py`
- `agent/planner.py`
- `agent/dispatcher.py`
- `agent/react_loop.py`
- SSE route in `api/routes/query.py` (`GET /query/stream`)
- `duckdb>=1.1.0` from `requirements.txt`

### 6.2 Additions

| Path | Responsibility |
|---|---|
| `agent/graph.py` | Single LangGraph `StateGraph` with nodes: `route`, `plan`, `execute_tools`, `synthesize`, `safety`, `respond`, `offline_answer`. |
| `api/routes/sync.py` | `GET /api/v1/sync/bundle`. |
| `models/sync.py` | Pydantic models: `SyncBundle`, `BundleRequest`, `BundleData`. |
| `offline/bundle_builder.py` | Builds gzipped district-scoped JSON bundles from seed data. |
| `tests/fakes/fake_gemma_client.py` | Deterministic fake for the Gemma 4 client used in all non-live tests. |
| `tests/fixtures/wheat_rust.jpg` | Small (~20 KB) real JPEG for multipart tests. |
| `tests/unit/test_graph.py` | Node-level tests for `agent/graph.py`. |
| `tests/unit/test_sync_bundle.py` | `bundle_builder.py` tests. |
| `tests/unit/test_error_envelope.py` | Error-shape tests. |
| `tests/integration/test_query_endpoint.py` | Updated to assert new response fields. |
| `tests/integration/test_image_upload.py` | Multipart round-trip. |
| `tests/integration/test_sync_endpoint.py` | Bundle endpoint incl. 304 handling. |
| `tests/integration/test_demo_smoke.py` | Must-pass-before-submission end-to-end. |
| `docs/frontend_handoff.md` | RN-specific guide: routing rules, thresholds, model ids, example requests. |

### 6.3 Modifications

- `api/routes/query.py` — add `POST /query/image`, accept `image_ref` in `POST /query`, remove SSE route.
- `models/api.py` — `AgentResponse` gains `confidence_score`, `model_used`, `fallback_hint`; unified `ErrorEnvelope`.
- `api/main.py` — register new sync router, drop rate-limit middleware.
- `agent/gemma_client.py` — target `gemma-4-26b-a4b-it` (primary) and `gemma-4-31b-it` (heavy) via `langchain-google-genai`.
- `agent/connectivity_router.py` — keep; used as a helper inside `graph.py`'s `route` node.
- `safety/layer.py` — add confidence-based escalation hook used by graph node (re-synthesize with 31B when conf < 0.70).
- `.env.example` — swap model ids, add new vars, remove rate-limit var.
- `docs/api_contract.md` — full rewrite for v0.2.
- `README.md`, `ARCHITECTURE.md`, `WRITEUP.md` — every Gemma 2 / Gemma 3 / "Gemma-class" reference replaced with explicit Gemma 4 variant id.
- `requirements.txt` — drop `duckdb`.

### 6.4 Module fidelity

All six intelligence modules remain registered; complexity trimmed where it doesn't serve the demo:

| Module | Status |
|---|---|
| `climate/` | Full: Open-Meteo live + Parquet offline fallback |
| `vision/` | Full: Gemma 4 multimodal via AI Studio |
| `scheme/` | Full: ChromaDB RAG |
| `crop_planner/` | Full: rule score + LLM narrative |
| `market/` | Trimmed: spot price + 7-day trend label only, no forecasting |
| `financial/` | Trimmed: KCC eligibility + ROI from pure Python rules |

### 6.5 Config (`.env.example`) after changes

```
GOOGLE_AI_STUDIO_KEY=

AI_STUDIO_MODEL=gemma-4-26b-a4b-it
AI_STUDIO_MODEL_HEAVY=gemma-4-31b-it

OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=gemma-4-e4b-it

CONNECTIVITY_MODE=auto
DATABASE_PATH=./data/krishisaathi.db
CHROMA_PATH=./data/chroma

CORS_ORIGINS=http://localhost:3000,http://localhost:5173

MAX_REACT_ITERATIONS=6
TOOL_TIMEOUT_SECONDS=8

SYNC_BUNDLE_CACHE_TTL_SECONDS=3600
IMAGE_UPLOAD_TTL_SECONDS=3600
IMAGE_MAX_MB=5
CONFIDENCE_THRESHOLD_LOW=0.70
```

## 7. Data flow (canonical sequences)

### 7.1 Online vision (crop disease photo)

1. RN app uploads image via `POST /query/image` → `image_ref`.
2. RN app calls `POST /query` with `image_ref`, `device_intent: "crop_disease"`.
3. Backend graph: `route` → online. `plan` (26B) emits tool calls. `execute_tools` runs `vision` (26B multimodal) + `climate`. `synthesize` (26B) writes Hindi response. `safety` validates.
4. `respond` assembles `AgentResponse` with `model_used: "gemma-4-26b-a4b-it"`.
5. Budget: p95 ≤ 8s end-to-end.

### 7.2 Offline weather (airplane mode)

1. RN app detects offline.
2. Intent classifier on-device = `weather`.
3. App reads cached `weather_history` from SQLite (stored from earlier `/sync/bundle`).
4. App calls Gemma 4 E4B via MediaPipe with system prompt + cached data.
5. App renders with "Offline · Gemma 4 E4B" badge. Backend uninvolved.

### 7.3 Online scheme RAG with escalation

1. RN app calls `POST /query` with `device_intent: "scheme_query"`.
2. Backend graph runs RAG via `scheme` module (ChromaDB top-5 chunks).
3. `synthesize` with 26B returns confidence 0.62.
4. `safety` node sees conf < 0.70 → re-synthesize with 31B → new conf 0.88.
5. Response `model_used: "gemma-4-31b-it"`, `tool_trace: ["scheme", "safety_escalation"]`.

### 7.4 Upstream rate limit degradation

1. AI Studio returns 429 to backend.
2. Backend retries once after 1s.
3. On second 429, backend returns `HTTP 429 { error: { code: "UPSTREAM_RATE_LIMIT", fallback_hint: "USE_ONDEVICE", retry_after_seconds: 30 } }`.
4. RN app reads `fallback_hint`, re-runs query on-device, shows "network busy" banner.

### 7.5 First-launch sync

1. RN app onboarding: user picks `state` + `district` (GPS auto-fill where possible, manual override always available).
2. App calls `GET /sync/bundle?state=Punjab&district=Ludhiana`.
3. Backend builds bundle on first request, caches 1h, returns ~1–2 MB gzipped JSON.
4. App writes to SQLite and stores `bundle_version` + expiry.
5. Subsequent launches send cached `bundle_version`; server returns 304 when unchanged.

## 8. Testing strategy

### 8.1 Coverage priorities

- **High:** `agent/graph.py` nodes, `safety/layer.py`, `offline/bundle_builder.py`.
- **Integration:** `POST /query`, `POST /query/image`, `GET /sync/bundle`, error envelope.
- **Light:** One happy-path test per trimmed module (`market`, `financial`).
- **None:** Pydantic validation, LLM text output, FastAPI middleware internals.

### 8.2 Infrastructure

- LLM mocking via `tests/fakes/fake_gemma_client.py` with pre-recorded JSON responses.
- ChromaDB: real, in-memory, ephemeral per session.
- HTTP: `respx` transport mock for Open-Meteo.
- Image: real small JPEG fixture.
- Async via `pytest-asyncio` (already in deps).

### 8.3 Must-pass submission test

`tests/integration/test_demo_smoke.py` exercises:
1. `GET /health` — asserts `gemma4_model_configured` starts with `"gemma-4-"`.
2. `GET /sync/bundle` for Punjab/Ludhiana — asserts non-empty schemes and prices.
3. `POST /query` for scheme question — asserts `model_used` starts with `"gemma-4-"`, `tool_trace` includes `"scheme"`, `data_source == "live"`.
4. `POST /query/image` + `POST /query` with `image_ref` — asserts `structured.kind == "disease"`.

Red on this test = do not submit.

### 8.4 TDD order

Strict red → green → commit loop. Each of the 22 commits in Section 9 is one TDD cycle.

### 8.5 Non-goals

No load tests, no security tests, no Playwright, no cross-repo contract tests, no mandatory live AI Studio tests in CI (`@pytest.mark.live` is opt-in).

## 9. Work breakdown (phases and commits)

| Phase | Commits | Duration |
|---|---|---|
| P0 — Foundation | 4 | ~0.5 day |
| P1 — Cleanup | 5 | ~1 day |
| P2 — Image upload | 3 | ~0.5 day |
| P3 — Sync bundle | 4 | ~1 day |
| P4 — Safety escalation | 2 | ~0.5 day |
| P5 — Docs & handoff | 3 | ~0.5 day |
| P6 — Submission ready | 1 | ~0.5 day |

### 9.1 Commit list

1. `test: add failing tests for new AgentResponse fields`
2. `feat: extend AgentResponse and response generator with Gemma 4 fields`
3. `test: standardize error envelope with retryable + fallback_hint`
4. `chore: swap .env.example + config to Gemma 4 variant ids`
5. `test: add failing tests for agent/graph.py nodes`
6. `refactor: consolidate agent files into agent/graph.py`
7. `chore: delete agent/orchestrator.py, planner.py, dispatcher.py, react_loop.py`
8. `chore: remove rate_limit middleware and drop duckdb dependency`
9. `chore: delete GET /query/stream SSE route`
10. `test: failing integration test for POST /query/image`
11. `feat: implement POST /query/image with 5MB cap and 1h TTL`
12. `feat: accept query.image_ref in POST /query`
13. `test: failing tests for offline/bundle_builder.py`
14. `feat: implement bundle_builder for schemes + mandi + calendar + weather_history`
15. `test: failing integration test for GET /sync/bundle`
16. `feat: implement sync route with 1h cache and bundle_version handling`
17. `test: failing test for safety node escalation (conf < 0.70 → 31b)`
18. `feat: implement 26b → 31b escalation in safety node`
19. `docs: rewrite docs/api_contract.md for v0.2`
20. `docs: add docs/frontend_handoff.md for RN team`
21. `docs: update README, ARCHITECTURE, WRITEUP with Gemma 4 variant ids`
22. `test: add test_demo_smoke.py covering all four demo beats`

## 10. Timeline (May 18 deadline)

| Window | Focus |
|---|---|
| Apr 20 – Apr 25 | Execute P0–P6 backend work |
| Apr 20 – May 5 (parallel) | RN team builds MediaPipe app |
| May 6 – May 10 | Integration testing with deployed backend |
| May 11 – May 13 | Field test with 5 farmers |
| May 14 – May 15 | Demo video, WRITEUP freeze, cover image |
| May 16 – May 17 | Kaggle dry run, identity verification, artifact upload |
| May 16 12:00 | Code freeze (48h before deadline) |
| May 18 23:59 UTC | Submit |

## 11. Deployment

| Need | Choice |
|---|---|
| Public URL | Render free tier or Fly.io free tier |
| Runtime | Docker container (existing `Dockerfile`) |
| Persistence | SQLite + Chroma on mounted volume |
| Secrets | Platform env vars (only `GOOGLE_AI_STUDIO_KEY` required) |
| Uptime | Best-effort; cold starts acceptable |

No Vertex AI, no Cloud Run, no Kubernetes.

## 12. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| AI Studio free-tier rate limits during judging | High | `fallback_hint: USE_ONDEVICE` + RN degradation |
| MediaPipe Gemma 4 integration slower than expected for RN team | Medium | Fallback to routing policy D (on-device offline-only); contract already supports it |
| ChromaDB first-boot index build >30s on judge machine | Medium | Ship pre-built Chroma index in `offline/data/chroma/` |
| AI Studio API breaking change before May 18 | Low | Pin `langchain-google-genai` version; run live smoke test in pre-submission checklist |
| Field test reveals comprehension issues | Medium | 2-day buffer (May 12–13); prompt-only iteration |
| UX bug discovered during video recording | Medium | Dry-run recording on May 11 |

## 13. Non-goals (explicit)

- No React Native code in this repo.
- No Gradio UI in this repo (RN app is the demo surface).
- No SSE streaming.
- No delta sync for offline bundle.
- No user-toggle "Prefer offline" setting.
- No auth beyond `X-Farmer-Id` header (optional for hackathon).
- No CORS changes (RN doesn't enforce CORS anyway).
- No ARIMA/Prophet forecasting in `market/`.
- No production observability stack.

## 14. Success criteria

1. `tests/integration/test_demo_smoke.py` passes against a fresh clone + `docker compose up --build`.
2. Every reference to an LLM variant in this repo names a specific Gemma 4 id (`gemma-4-26b-a4b-it`, `gemma-4-31b-it`, `gemma-4-e4b-it`, `gemma-4-e2b-it`).
3. `docs/frontend_handoff.md` exists and the RN team confirms it's enough to build against.
4. A 3-minute demo video can be produced showing the four beats (health, sync, online scheme, online vision) using this backend plus the RN app.
5. Public backend deployed on a free tier with a stable URL linked from `README.md`.

---

**Document version:** 0.1 (initial draft)
**Next step:** user reviews this spec; on approval, the `writing-plans` skill produces the executable implementation plan.
