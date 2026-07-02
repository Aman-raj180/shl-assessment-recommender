# SHL Assessment Recommender

Conversational agent that turns a vague hiring intent into a grounded
shortlist of SHL Individual Test Solutions, via a stateless FastAPI `/chat`
endpoint. Built for the SHL AI Intern take-home assignment.

## Stack
- **FastAPI** — API layer (`/health`, `/chat`)
- **Groq (LLaMA 3.3 70B, `llama-3.3-70b-versatile`)** — conversational
  reasoning, decides clarify / recommend / refine / compare / refuse each
  turn, using JSON mode for structured output. Chosen over Gemini's free
  tier because Groq's free rate limit is per-minute with no daily ceiling
  and needs no billing/card — important because the automated evaluator
  runs many 8-turn conversations back to back, and a daily-capped free
  tier risks getting blocked mid-evaluation.
- **BM25** (`rank_bm25`) — keyword retrieval over the catalog. Chosen over
  dense embeddings because the catalog is small (370 items) and queries
  contain exact product/technical terms; also avoids downloading model
  weights at container start, which matters for cold starts on free hosting.
- No database — the API is stateless per the assignment spec; the catalog
  and BM25 index are just loaded into memory once at process start.

## Project layout
```
app/
  main.py          FastAPI app, /health and /chat routes
  agent.py         Orchestration: retrieval -> prompt -> LLM -> validated response
  llm_client.py    Groq SDK wrapper, JSON-mode output, retry/fallback, mock mode
  retrieval.py     BM25 index + catalog lookups
  schemas.py       Pydantic request/response models matching the API contract
  catalog_clean.json  370 Individual Test Solutions (Job Solution bundles excluded)
data/
  clean_catalog.py  One-off script: raw scrape -> catalog_clean.json
  raw_catalog.json  Original scrape provided by SHL
tests/
  run_traces.py     Replays the 10 sample conversation traces against a running server
sample_conversations/GenAI_SampleConversations/  Provided sample traces (C1.md..C10.md)
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Set your Groq API key (free, get one at https://console.groq.com/keys):

```bash
export GROQ_API_KEY="your-key-here"     # Windows (PowerShell): $env:GROQ_API_KEY="your-key-here"
```

## Run locally

```bash
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Check it's up:
```bash
curl http://localhost:8000/health
```

## Run the sample-trace test harness

In a second terminal, with the server still running:

```bash
pip install requests
python3 tests/run_traces.py --base-url http://localhost:8000 --traces-dir sample_conversations/GenAI_SampleConversations --delay 2
```

This replays all 10 provided traces turn-by-turn and prints, per trace:
recommendation count, `end_of_conversation`, a rough recall score against
the trace's expected shortlist, and the actual reply/recommendations —
useful for spotting bad clarify loops, wrong recommendations, or schema
issues before deploying. `--delay` paces requests to stay under Groq's
free-tier per-minute rate limit during local testing.

Note: this harness replays the *literal* user turns recorded in each trace.
The real evaluator uses an LLM-simulated user that can answer out of order,
refuse to answer, or self-correct — so treat this as a development sanity
check, not the final score.

## Mock mode (no API key needed)

For pipeline smoke-testing without any API calls:
```bash
export MOCK_LLM=1
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
`MOCK_LLM=1` short-circuits the LLM call with deterministic dummy logic so
you can verify retrieval, schema, and FastAPI wiring in isolation.

## Deployment

See `DEPLOY.md`.
