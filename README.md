# KrishiSaathi AI — Backend

Hybrid **Gemma 4** farmer assistant API for the Gemma 4 Good Hackathon.

- **Backend primary:** `gemma-4-26b-a4b-it` (Google AI Studio free tier)
- **Backend escalation:** `gemma-4-31b-it` (Google AI Studio free tier)
- **On-device (separate RN app):** `gemma-4-e4b-it` / `gemma-4-e2b-it` via MediaPipe LLM Inference

Features: **LangGraph** agent loop, **Open-Meteo** weather, **ChromaDB** scheme RAG, **SQLite** persistence, district-scoped **offline sync bundle**, multipart image upload.

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

## License

See [LICENSE](LICENSE).
