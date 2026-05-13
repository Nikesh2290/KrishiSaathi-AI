# KrishiSaathi AI — Gemma 4 Good Hackathon submission

## Hook

KrishiSaathi puts a **Gemma 4** agronomist in the pocket of India's 100M smallholder farmers — in their language, and when the network is gone.

## Problem

Smallholder farmers face fragmented information: weather risk, crop diseases, mandi prices, and government schemes live across apps, PDFs, and offices. Connectivity and literacy barriers make cloud-only chatbots unusable.

## Solution

KrishiSaathi is a **hybrid on-device + cloud** AI agent:

- **On-device** (separate RN app, this repo's sibling): `gemma-4-e4b-it` via **MediaPipe LLM Inference** — private, offline, Hindi/Hinglish, runs on mid-range Android.
- **Online** (this backend): `gemma-4-26b-a4b-it` via **Google AI Studio free tier**, with **escalation to `gemma-4-31b-it`** for low-confidence queries.
- **Offline sync bundle**: district-scoped ~1–2 MB gzipped JSON (schemes, mandi prices, crop calendar, 5-year weather averages) — the RN app downloads it once and keeps working without a signal.

## Why Gemma 4

- Four Apache 2.0 variants from edge (E2B/E4B) to server (26B MoE, 31B dense) — one model family covers the whole stack.
- E4B runs on 4 GB RAM phones — real-world Indian device median.
- 140+ languages, native multimodal — Hindi voice + leaf photos work out of the box.

## Impact

- **Digital equity**: works in airplane mode in a Ludhiana field.
- **Climate resilience**: weather + crop-plan advice from cached data alone.
- **Responsible AI**: safety layer strips unsourced pesticide dosages; escalates low-confidence answers to the 31B model before the farmer sees them.

## Architecture

```
RN app (MediaPipe, Gemma 4 E4B/E2B)
  ├── offline intents → on-device
  └── online intents → POST /api/v1/query/stream → this backend
                                              ├── LangGraph StateGraph
                                              │   (route → plan → tools → synth → safety → respond)
                                              ├── Gemma 4 26B A4B on AI Studio
                                              │   ↳ escalates to 31B if confidence < 0.70
                                              └── Tools: climate, vision, scheme-RAG,
                                                         market, crop-planner, financial
```

Full design: [ARCHITECTURE.md](ARCHITECTURE.md). API contract: [docs/api_contract.md](docs/api_contract.md). RN handoff: [docs/frontend_handoff.md](docs/frontend_handoff.md).

## Demo

3-minute video covering:

1. `GET /health` — backend alive, `gemma-4-26b-a4b-it` configured.
2. Online scheme query in Hindi → `gemma-4-26b-a4b-it` answers with citations.
3. Photo of diseased wheat leaf → `POST /query/image` → `POST /query` → structured disease card.
4. Airplane mode → RN app answers weather + crop plan offline via on-device `gemma-4-e4b-it`.
5. Safety layer stripping an unverified pesticide dosage (20-second close-up).

## Field test

Usefulness self-reported by 5 farmers in Ludhiana (Apr 2026): **[fill from field test]**.

## Repo & license

Apache 2.0. See [LICENSE](LICENSE).
