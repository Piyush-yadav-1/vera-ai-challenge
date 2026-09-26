# Vera AI Engine — Magicpin AI Challenge Submission

## Overview

**Vera AI Engine** is an automated merchant communication agent built for the Magicpin AI Challenge. It operates as a FastAPI server that receives context pushes (categories, merchants, customers, triggers) and composes personalized WhatsApp messages using LLM-powered composition grounded in real merchant data.

## Architecture

```
Judge Harness ──HTTP/JSON──► Vera Engine (FastAPI)
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼             ▼
              Rule-Based    LLM Composer   Context Store
              Router        (OpenRouter)   (In-Memory)
              ├─ Auto-reply    │
              ├─ Hostile       │
              └─ Intent        │
                               ▼
                         Structured JSON Response
```

### Key Design Decisions

1. **Rule-based routing first, LLM second**: Auto-reply, hostile, and intent-commitment messages are detected via regex patterns before invoking the LLM. This ensures deterministic behavior for critical edge cases (auto-reply loops, hostile exits, intent transitions) and avoids burning LLM tokens on pattern-matchable inputs.

2. **Schema compliance**: The bot returns `{"action": "send"|"end"|"wait", "body": "..."}` for reply endpoints and `{"actions": [...]}` for tick, matching the judge harness contract exactly.

3. **Fail-loud over fail-silent**: LLM failures raise HTTP 502 errors instead of returning static fallback text. This ensures the judge sees real failures rather than scoring hardcoded responses.

4. **Retry with backoff**: Rate-limited (429) LLM calls are retried with exponential backoff (up to 3 attempts) before failing.

5. **Grounded composition**: The system prompt enforces strict grounding — every claim must reference data from the pushed context. No fabricated offers, stats, or citations.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/v1/healthz` | Liveness probe |
| `GET` | `/v1/metadata` | Bot identity & approach |
| `POST` | `/v1/context` | Receive context pushes (category, merchant, customer, trigger) |
| `POST` | `/v1/tick` | Compose proactive messages from available triggers |
| `POST` | `/v1/reply` | Handle merchant/customer replies in conversation |

## How to Run Locally

### Prerequisites
- Python 3.10+
- An OpenRouter API key (free tier works)

### Setup

```bash
# Clone and enter directory
cd vera-ai-challenge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Set your API key (or edit main.py directly)
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"

# Start the server
uvicorn main:app --host 0.0.0.0 --port 8080
```

### Run the Judge Simulator

```bash
# In a second terminal, with the server running:
python judge_simulator.py
```

## Model

- **Primary**: `qwen/qwen3.8-27b:free` via OpenRouter (dense 27B, strong structured output)
- **Fallback alternatives**: `nvidia/nemotron-3-super:free`, `nvidia/nemotron-3.5-lightning:free`
- Temperature: 0.0 (deterministic)

## Deployment

For public HTTPS deployment (Render / Railway):

```bash
# Render — set these environment variables:
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=qwen/qwen3.8-27b:free

# Start command:
uvicorn main:app --host 0.0.0.0 --port $PORT
```

## Tradeoffs

- **Free-tier LLM**: Using a free OpenRouter model means occasional rate limits (429s). The retry logic with exponential backoff mitigates this, but under heavy load some ticks may return empty action lists.
- **In-memory state**: Context is stored in a Python dict — restarting the server loses all pushed context. For production, this would be backed by Redis or a database.
- **Rule-based edge cases**: Auto-reply, hostile, and intent detection use regex patterns rather than LLM classification. This is faster and more deterministic but less flexible for novel phrasings.

## Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI server — all endpoints and LLM composition logic |
| `judge_simulator.py` | Local test harness (provided by Magicpin) |
| `requirements.txt` | Python dependencies |
| `dataset/` | Base dataset (categories, merchants, customers, triggers) |
| `challenge-brief.md` | Challenge specification |
| `challenge-testing-brief.md` | API contract and testing specification |

## Author

Piyush Yadav
