---
title: KrishiSaathi AI Backend
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# KrishiSaathi AI — Backend

Hybrid **Gemma 4** farmer assistant API for the Gemma 4 Good Hackathon.

- **Backend primary:** `gemma-4-26b-a4b-it` (Google AI Studio free tier)
- **Backend escalation:** `gemma-4-31b-it` (Google AI Studio free tier)
- **On-device (separate RN app):** `gemma-4-e4b-it` / `gemma-4-e2b-it` via MediaPipe LLM Inference

Features: **LangGraph** agent loop, **Open-Meteo** weather, **ChromaDB** scheme RAG, **SQLite** persistence, district-scoped **offline sync bundle**, multipart image upload.

**Optional:** **Supabase** for online Auth (email/password → stable `farmer_id` UUID), **Postgres** for twin + query history, **pgvector** for scheme retrieval. Apply [`db/migrations/supabase_schema.sql`](db/migrations/supabase_schema.sql) in the Supabase SQL editor, then set `SUPABASE_*` vars in `.env`. Offline mode still uses SQLite + Chroma; `POST /api/v1/sync/push` (or startup) uploads unsynced rows and scheme embeddings.

## Quickstart

1. **Python 3.11+**.

2. **Environment**

   ```bash
   cp .env.example .env
   # Set GOOGLE_AI_STUDIO_KEY (https://aistudio.google.com/app/apikey)
   ```

3. **Install & run**

   ```bash
   pip install -r requirements.txt
   python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Docs**

   - Swagger: <http://localhost:8000/docs>
   - Contract: [docs/api_contract.md](docs/api_contract.md)
   - RN handoff: [docs/frontend_handoff.md](docs/frontend_handoff.md)
   - Architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
   - Hackathon write-up: [WRITEUP.md](WRITEUP.md)

5. **Offline seed data** is created on first startup (`offline/bootstrap_data.py`).

## Tests

```bash
python -m pytest tests/ -v
python -m pytest tests/integration/test_demo_smoke.py -v  # must pass before submission
```

## Docker

```bash
docker compose up --build
```

## Hugging Face Spaces (Docker)

This repo’s [`Dockerfile`](Dockerfile) runs Uvicorn on port **7860** (`api.main:app`). Logs go to **stdout** (visible in the Space **Logs** tab) and, by default in the image, also to a **rotating file** at `./logs/app.log` inside the container (best-effort; not guaranteed to persist across rebuilds).

**Space variables (optional overrides)**

| Variable | Default in image | Purpose |
|----------|------------------|---------|
| `LOG_LEVEL` | `INFO` | Root / uvicorn log level |
| `LOG_JSON` | `true` | Structured JSON lines |
| `LOG_FILE` | `./logs/app.log` | Set empty to disable file logging |

Set `GOOGLE_AI_STUDIO_KEY` and any other secrets in the Space **Settings → Variables and secrets**.

**Verify after deploy**

- Call `GET /api/v1/health` and check Space Logs for a request line; the response should include **`X-Request-Id`** (or echo your incoming `X-Request-Id` header).
- Unhandled errors should include a stack trace in logs (see `models/errors.py`).

## License

See [LICENSE](LICENSE).
