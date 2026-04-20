# KrishiSaathi AI — Gemma 4 Good Hackathon write-up (template)

## Problem

Smallholder farmers in India face fragmented information: weather risk, crop diseases, mandi prices, and government schemes are spread across apps, PDFs, and offices. Connectivity and literacy barriers make generic cloud-only chatbots unusable.

## User story

**Ramesh Kumar** (Punjab) grows wheat and mustard. He needs actionable guidance in **Hindi**, sometimes from a **photo** of diseased leaves, and answers when the network is **unreliable**.

## Solution

**KrishiSaathi** is a **backend-only** AI agent API that:

1. Routes between **local Ollama (Gemma-class)** and **Google AI Studio (Gemini API)** for planning and vision.
2. Uses a **LangGraph**-style pipeline: route → plan tools → execute tools → synthesize answer → **safety** checks.
3. Integrates **free** data sources: **Open-Meteo** (weather), **offline CSV/Parquet/JSON** (mandi, weather history, schemes), **ChromaDB** (scheme RAG).
4. Exposes a clean **REST + SSE** contract for any frontend (mobile / web).

## Why Gemma / Gemma-class

- **Open weights & local deploy** via Ollama — aligns with low-connectivity and sponsor emphasis on edge tools.
- **Multimodal** path for crop-disease photos (vision prompt → structured JSON).
- **Tool use** via an explicit planner JSON + dispatcher (auditable `tool_trace` in every response).

## Impact

- Targets **digital equity** and **climate/agriculture resilience** (weather + offline fallbacks).
- **Safety layer** reduces unsourced medical-style dosages and flags low-confidence vision.

## Technical summary

- **Stack:** FastAPI, SQLite, ChromaDB, LangGraph, httpx, DuckDB.
- **Run:** see [README.md](README.md).
- **API:** [docs/api_contract.md](docs/api_contract.md) and `/docs` Swagger.

## Demo

Record a short video showing: health check → POST query (Hindi scheme question) → optional SSE stream. Frontend can be built separately against this API.

## Repo & license

See [LICENSE](LICENSE).
