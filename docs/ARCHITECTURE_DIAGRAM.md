# Architecture Flow Diagram

Up-to-date end-to-end architecture of the Multi-Agent SQL Analytics system,
reflecting the current state: unified database, JWT auth, and correlated
structured logging.

The Mermaid blocks below render natively on GitHub (and in VS Code with the
Mermaid extension). Pre-rendered **PNG exports** are in
[`docs/diagrams/`](./diagrams/) for sharing without GitHub — see
[Rendered exports](#rendered-exports-png) at the bottom. To edit, paste any
block into <https://mermaid.live> or import this file into Eraser/draw.io.

---

## 1. System overview (request lifecycle)

```mermaid
flowchart TD
    subgraph Client["🖥️ Browser (React + Vite SPA)"]
        UI["SQL Analytics UI"]
        LoginUI["Login screen<br/>(AuthContext, token in localStorage)"]
    end

    subgraph Edge["🌐 DigitalOcean droplet — arshman.techquest.ai"]
        NGINX["nginx :443 (HTTPS / certbot)<br/>security headers (HSTS, CSP, …)"]
        FE["PM2: sql-agents-frontend<br/>serve dist/ :8011"]
        API["PM2: sql-agents-api<br/>uvicorn (FastAPI) :8010"]
    end

    UI -->|"static assets /"| NGINX
    NGINX -->|"/"| FE
    UI -->|"POST /api/v1/* (+ Bearer JWT)"| NGINX
    NGINX -->|"/api/ → :8010"| API

    LoginUI -->|"POST /api/v1/auth/login"| NGINX

    API --> AUTH["Auth layer (JWT)"]
    API --> PIPE["Analytics pipeline"]
    API --> LOGS["Structured JSON logs<br/>logs/pipeline.log (request_id)"]
```

---

## 2. Authentication (JWT)

```mermaid
sequenceDiagram
    participant B as Browser
    participant API as FastAPI
    participant DB as analytics.db (users)

    B->>API: POST /api/v1/auth/login {username, password}
    API->>DB: SELECT password_hash WHERE username=? (parameterised)
    DB-->>API: bcrypt hash
    API->>API: verify_password() + sign HS256 JWT (60 min)
    API-->>B: { access_token, token_type, expires_in }
    Note over B: token stored in localStorage

    B->>API: POST /api/v1/query  (Authorization: Bearer <token>)
    API->>API: get_current_user() verifies signature + expiry
    alt valid
        API-->>B: 200 — SSE stream
    else missing/invalid/expired
        API-->>B: 401 → frontend clears token, shows login
    end
```

Public endpoints: `GET /health`, `POST /auth/login`.
Protected (require Bearer token): `POST /query`, `POST /compare`, `POST /build-index`, `GET /auth/me`.

---

## 3. The four-phase analytics pipeline

```mermaid
flowchart TD
    REQ["POST /api/v1/query<br/>(authenticated)"] --> RID["new_request_id()<br/>stamps every log line"]
    RID --> SSE["SSE stream (asyncio.Queue → StreamingResponse)"]

    subgraph P1["① Planning (LLM)"]
        PO["ParentOrchestrator<br/>identify domains"]
        CP1["ChildTaskPlanner: airport"]
        CP2["ChildTaskPlanner: tech_startup"]
        CP3["ChildTaskPlanner: restaurant"]
        PO -->|asyncio.gather| CP1 & CP2 & CP3
        CP1 & CP2 & CP3 --> DAG["Merged global DAG<br/>(TaskNodes, domain-prefixed ids)"]
    end

    subgraph P2["② DAG execution (streaming, event-gated)"]
        EX["DAGExecutor<br/>one coroutine per task,<br/>awaits only its direct deps"]
        SQLT["sql → SQLAgent"]
        DERT["derived → compute_derived()<br/>(pure Python aggregation)"]
        PLOTT["plot → PlotAgent"]
        EX --> SQLT & DERT & PLOTT
    end

    subgraph P3["③ Blackboard coordination (dedup + cache)"]
        BB["run_with_blackboard()<br/>normalise SQL → sha256(domain+sql)"]
        CACHE{"result cache hit?"}
        OWNER["SETNX owner → execute → publish"]
        SUBS["subscriber → await pub/sub"]
        BB --> CACHE
        CACHE -->|yes| RET["return cached rows"]
        CACHE -->|no| OWNER
        CACHE -->|no, claimed| SUBS
    end

    subgraph P4["④ Synthesis (LLM)"]
        SYN["SynthesisAgent<br/>plain-English answer + numbers"]
    end

    SSE --> P1 --> P2
    SQLT --> P3
    OWNER --> DBQ["execute_query(sql, domain)"]
    P2 --> P4 --> OUT["synthesis_complete event<br/>answer + charts + stats"]
    OUT --> SSE

    RID -.->|request_id on all logs| LOG["logs/pipeline.log (JSON)"]
```

---

## 4. Schema retrieval & the unified database

```mermaid
flowchart LR
    SA["SQLAgent (per domain)"] --> RT["SchemaRetriever<br/>keyword match → FAISS fallback"]
    RT --> EMB["SentenceTransformer<br/>all-MiniLM-L6-v2 (shared singleton)"]
    RT --> IDX["FAISS index per domain<br/>(prefixed view names)"]
    SA --> GEN["LLM generates SQL<br/>(Groq llama-3.3-70b)"]
    GEN --> GW["execute_query(sql, domain)<br/>Table Gateway"]

    subgraph UDB["🗄️ db/analytics.db (single file)"]
        direction TB
        BASE["Base tables (domain column):<br/>employees · flights · projects · menus"]
        V1["VIEW airport_employees / airport_flights"]
        V2["VIEW tech_startup_employees / _projects"]
        V3["VIEW restaurant_employees / _menus"]
        USERS["users (bcrypt hashes)"]
        BASE --> V1 & V2 & V3
    end

    GW --> V1 & V2 & V3
    note["Agents query VIEWS only →<br/>domain isolation enforced at the DB layer"]
    V1 -.- note
```

---

## 5. Deployment & CI/CD

```mermaid
flowchart LR
    DEV["git push → main<br/>(backend / frontend)"] --> GH["GitHub Actions"]
    GH --> CI["CI: ruff lint + import checks<br/>+ pure-Python unit tests"]
    GH --> CD["CD: rsync over SSH<br/>(dedicated revocable deploy key)"]
    CD --> DROP["Droplet"]
    DROP --> MIG["db/migrate.py (idempotent)<br/>ensure users table + seed admin"]
    DROP --> IDXB["build FAISS index if missing /<br/>on unified-DB creation"]
    DROP --> PM2["pm2 restart --update-env"]
    PM2 --> HC["health check /api/v1/health"]
    note2["Droplet holds no GitHub creds;<br/>.env / DB / index are droplet-only state"]
    DROP -.- note2
```

---

## Key design properties

| Concern | How it's handled |
|---|---|
| **DB access pattern** | Table Gateway (`db/pool.py`) — one `execute_query(sql, domain)` routes to `analytics.db`; raw SQL, no ORM. |
| **Domain isolation** | Per-domain SQL **views** in one file; agents query views, so a domain agent cannot read another domain's rows. |
| **Auth** | JWT (HS256), bcrypt-hashed users table; protected routes via `Depends(get_current_user)`. |
| **Dedup / caching** | Blackboard: `sha256(domain+canonical_sql)`, SETNX ownership, pub/sub, TTL heartbeat. |
| **Schema context** | FAISS semantic retrieval returns only relevant tables → fewer tokens, fewer hallucinations. |
| **Observability** | Correlated `request_id` on every log line (ContextVar) + JSON rotating file. |
| **Streaming** | SSE via `asyncio.Queue` → `StreamingResponse`; client-disconnect cancels the pipeline task. |
| **CI/CD** | Push-based GitHub Actions; lint+import CI; rsync deploy with idempotent migrations + health gate. |

---

## Rendered exports (PNG)

Pre-rendered images of the diagrams above (generated with mermaid-cli):

1. [System overview](./diagrams/01-system-overview.png)
2. [Authentication (JWT)](./diagrams/02-authentication.png)
3. [Four-phase pipeline](./diagrams/03-pipeline.png)
4. [Schema retrieval & unified database](./diagrams/04-schema-and-database.png)
5. [Deployment & CI/CD](./diagrams/05-deployment-cicd.png)
