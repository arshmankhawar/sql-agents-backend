# Multi-Agent SQL System — Blackboard Architecture

A production-grade implementation of the redesigned multi-agent SQL pipeline with:

- **Zero duplicate DB queries** — atomic SETNX ownership + shared result cache
- **In-flight coordination** — waiting agents subscribe via Redis Pub/Sub
- **Crash-safe** — TTL + heartbeat prevents dead-agent stalls
- **Smart schema retrieval** — FAISS semantic search serves only relevant tables per agent
- **Plot Agent isolation** — never queries PostgreSQL, reads Blackboard only
- **Unlimited agent scalability** — deduplication holds at 3, 20, or 50 agents

---

## Architecture

```
Client
   │
   ▼
Agent Orchestrator
   │
   ▼
Global Task Planner ◄── Groq LLM (task decomposition → DAG)
   │
   ▼  (asyncio parallel DAG)
┌─────────────────────────────────────────────────────┐
│                  BLACKBOARD (Redis)                  │
│  • Query Registry   (SETNX atomic ownership + TTL)  │
│  • In-Flight Tracker (heartbeat renewal)            │
│  • Shared Result Cache (Hash, long TTL)             │
│  • Pub/Sub Channels  (per query_hash notification)  │
└─────────────────────────────────────────────────────┘
        │              │              │
        ▼              ▼              ▼
     SQL A          SQL B          SQL C
        │              │              │
        └──────────────┴──────────────┘
                       │
            Schema Retrieval Layer
         (FAISS + sentence-transformers)
                       │
              asyncpg Connection Pool
                       │
              PostgreSQL (read-only)

Plot Agent ──► reads Blackboard only (no DB access)
```

---

## Quick Start

### 1. Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11+ |
| Docker Desktop | Latest |
| Groq API Key | Free at [console.groq.com](https://console.groq.com) |

### 2. Start Redis

```powershell
docker run -d --name redis-blackboard -p 6379:6379 redis:latest
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure Environment

```powershell
copy .env.example .env
```

Edit `.env` and set your Groq API key:

```env
GROQ_API_KEY=gsk_your_key_here
```

### 5. Build the Schema Index

```powershell
python main.py --build-index
```

This creates the FAISS vector index from the mock schema (or your real PostgreSQL schema if `USE_MOCK_DB=false`).

---

## Running the System

### Full Pipeline Demo (requires Groq API key)

```powershell
python main.py
```

Uses the default request: *"Show me revenue trends by region and top customers by lifetime value for 2025, then plot both."*

Custom request:

```powershell
python main.py --request "What are the top 5 products by revenue and inventory levels?"
```

### Deduplication Demo (no API key needed)

Demonstrates the core deduplication guarantee: 5 concurrent agents request the same SQL — only **1 DB execution** occurs.

```powershell
python main.py --dedup-demo
```

Expected output:
```
── Results ──────────────────────────────────────────────────────────
  agent_alpha          source=owner        rows=20    52ms
  agent_beta           source=subscriber   rows=20    54ms
  agent_gamma          source=subscriber   rows=20    54ms
  agent_delta          source=subscriber   rows=20    55ms   ← normalised to same hash
  agent_epsilon        source=subscriber   rows=20    55ms

── Summary ──────────────────────────────────────────────────────────
  DB executions:  1  (should be 1)
  Subscribers:    4
  Cache hits:     0
```

---

## Project Structure

```
SQL Agents Task/
├── blackboard/
│   ├── client.py           # Async Redis connection singleton
│   ├── query_registry.py   # SETNX claim, TTL, Pub/Sub notify/subscribe
│   └── result_cache.py     # Shared result store (get/set/delete)
├── schema/
│   ├── indexer.py          # FAISS index builder (mock or real PG schema)
│   └── retriever.py        # Semantic top-k table retrieval per agent task
├── agents/
│   ├── base_agent.py       # Blackboard-aware execution (all agents inherit)
│   ├── sql_agent.py        # LLM SQL generation + Blackboard execution
│   └── plot_agent.py       # Blackboard consumer only — no DB access
├── planner/
│   └── task_planner.py     # LLM task decomposition → validated DAG
├── dag/
│   └── executor.py         # Parallel DAG execution engine
├── db/
│   └── pool.py             # asyncpg pool + rich mock DB
├── utils/
│   └── sql_normalizer.py   # sqlglot normalisation + SHA-256 hashing
├── config.py               # Centralised config from .env
├── main.py                 # Entry point (3 run modes)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Connecting a Real PostgreSQL Database

1. Set in `.env`:
   ```env
   USE_MOCK_DB=false
   POSTGRES_DSN=postgresql://user:password@host:5432/dbname
   ```

2. Rebuild the schema index:
   ```powershell
   python main.py --build-index
   ```

The system introspects `information_schema.columns` to build the FAISS index automatically.

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Groq API key (required for full pipeline) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `QUERY_TTL_SECONDS` | `120` | In-flight entry expiry (crash safety) |
| `RESULT_CACHE_TTL_SECONDS` | `3600` | How long completed results are cached |
| `USE_MOCK_DB` | `true` | Use built-in mock data (no real PG needed) |
| `POSTGRES_DSN` | — | Real PostgreSQL DSN |
| `SCHEMA_INDEX_PATH` | `./schema_index` | FAISS index directory |
| `SCHEMA_TOP_K` | `5` | Max tables retrieved per agent task |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |

---

## How Each Problem Is Solved

| Problem | Solution |
|---|---|
| Duplicate queries | SHA-256 hash of normalised SQL → SETNX atomic claim → one owner executes, rest subscribe |
| No agent awareness | Blackboard is single source of truth — all agents read/write the same state |
| Excessive schema context | FAISS semantic search returns only relevant tables per task |
| Poor scalability | Deduplication is O(1) regardless of agent count |
| Plot Agent re-querying | PlotAgent reads exclusively from result cache — `execute_query` is never called |
