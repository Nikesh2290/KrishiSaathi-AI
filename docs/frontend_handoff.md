# KrishiSaathi AI — Frontend Handoff (for the React Native repo)

This document is the contract between the backend (this repo) and the RN app repo for the Gemma 4 Good Hackathon.

## 1. Model map

| Layer | Variant | Where |
|---|---|---|
| Backend primary | `gemma-4-26b-a4b-it` | Google AI Studio |
| Backend heavy (escalation) | `gemma-4-31b-it` | Google AI Studio |
| On-device primary | `gemma-4-e4b-it` | MediaPipe LLM Inference (Android) |
| On-device low-RAM | `gemma-4-e2b-it` | MediaPipe LLM Inference |

All four are Gemma 4 (Apache 2.0, released 2026-04-02).

## 2. Routing rules (intent-based)

```
if connectivity == "offline":
    use on-device gemma-4-e4b-it (or e2b on <4GB RAM)
elif intent in {"weather", "crop_plan", "general", "alert"}:
    use on-device (even when online — latency + data saver)
else:
    POST /api/v1/query  (backend gemma-4-26b-a4b-it)
```

## 3. Confidence threshold

```ts
const CONFIDENCE_THRESHOLD_LOW = 0.70;
```

If an on-device response has `confidence < 0.70`, render the answer with a non-blocking CTA: "Get expert analysis (needs internet)". On tap + online, call `POST /api/v1/query` with the same query.

## 4. Fallback on backend errors

Always read `error.fallback_hint`:

- `USE_ONDEVICE` → silently re-run on-device; show a "network busy" banner.
- `RETRY_ONLINE_LATER` → show a retry CTA; do not re-run on-device.
- `null` → show the error message; no auto-action.

## 5. Image flow

1. `POST /api/v1/query/image` (multipart) → `{ image_ref, expires_at }`.
2. `POST /api/v1/query` with `query.image_ref` set. Do not send base64 in `/query`.

`image_ref` TTL is 1 hour. On 404 `IMAGE_REF_EXPIRED`, re-upload.

## 6. First-launch sync

During onboarding, collect `state` + `district` (prefill from GPS; always allow manual).

```
GET /api/v1/sync/bundle?state=Punjab&district=Ludhiana
```

- Response is gzipped JSON; read `Content-Encoding: gzip`.
- Persist the bundle to SQLite (`op-sqlite` or `expo-sqlite`).
- Store `bundle_version` and re-check on launch. If unchanged, server returns 304.

## 7. On-device model download

The RN app downloads Gemma 4 E4B / E2B weights from Google AI Edge Gallery on first launch; the backend does not serve model weights.

## 8. Error envelope (all non-2xx)

```ts
type ErrorEnvelope = {
  error: {
    code: string;
    message: string;
    retryable: boolean;
    retry_after_seconds?: number;
    fallback_hint?: "USE_ONDEVICE" | "RETRY_ONLINE_LATER" | null;
  };
};
```

See `docs/api_contract.md` Section 6 for the full error code table.

## 9. Latency budgets (app timeout targets)

| Endpoint | p95 target | App timeout |
|---|---|---|
| `POST /query` | 6 s | 10 s |
| `POST /query/image` | 2 s | 5 s |
| `GET /sync/bundle` | 5 s | 15 s |
| `GET /health` | 500 ms | 3 s |

## 10. Example requests (TypeScript fetch)

```ts
// Online scheme query
await fetch(`${BASE}/api/v1/query`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    farmer_id: farmerId,
    query: { text, language: "hi" },
    context: {
      location: { state, district },
      connectivity: "online",
      device_intent: "scheme_query",
      device_capabilities: { ondevice_model: "gemma-4-e4b-it" },
    },
  }),
});

// Image upload
const fd = new FormData();
fd.append("image", { uri, type: "image/jpeg", name: "leaf.jpg" } as any);
fd.append("farmer_id", farmerId);
fd.append("purpose", "crop_disease");
await fetch(`${BASE}/api/v1/query/image`, { method: "POST", body: fd });
```
