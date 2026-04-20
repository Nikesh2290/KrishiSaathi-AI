# KrishiSaathi AI — Backend

Hybrid **edge + cloud + offline** farmer assistant API: **Gemma-class** models (Ollama + Google AI Studio), **LangGraph** tool loop, **Open-Meteo** weather, **ChromaDB** scheme RAG, **SQLite** persistence.

## Quickstart

1. **Python 3.11+** and (optional) **[Ollama](https://ollama.com/)** with a small model, e.g. `ollama pull gemma2:2b`.

2. **Environment**

   ```bash
   copy .env.example .env
   ```

   Set `GOOGLE_AI_STUDIO_KEY` if you want cloud fallback when Ollama is down (get a key from [Google AI Studio](https://aistudio.google.com/app/apikey)).

3. **Install & run**

   ```bash
   pip install -r requirements.txt
   python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Docs**

   - Swagger: <http://localhost:8000/docs>
   - Human-readable contract: [docs/api_contract.md](docs/api_contract.md)
   - Postman: [docs/postman_collection.json](docs/postman_collection.json)

5. **Offline seed data** is created on first startup (`offline/bootstrap_data.py`). Rebuild anytime:

   ```bash
   python -m offline.bootstrap_data
   ```

## Tests

```bash
python -m pytest tests/ -v
```

## Docker

```bash
docker compose up --build
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md).

## Hackathon

See [WRITEUP.md](WRITEUP.md) for a Kaggle / Gemma 4 Good–style submission template.
