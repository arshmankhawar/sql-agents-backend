# DEPLOYMENT.md — Backend (sql-agents-backend)

> **Read this file fully before deploying.** It contains everything needed to deploy
> this FastAPI backend to DigitalOcean, including project-specific gotchas that are NOT
> obvious from the code.

---

## 1. What this service is

A FastAPI application (multi-agent SQL analytics pipeline) that streams results over
Server-Sent Events (SSE). It is the **backend half** of a two-repo project:

- **This repo** (`sql-agents-backend`) → deploys as a **DigitalOcean App Platform Web Service**.
- Companion repo (`sql-agents-frontend`) → deploys as a **DigitalOcean Static Site**.

The frontend calls this backend at routes prefixed **`/api/v1`** (e.g. `POST /api/v1/compare`,
`POST /api/v1/query`).

---

## 2. Critical project-specific facts (READ THESE FIRST)

These are the things that will break a naive deployment. Address every one.

### 2a. Generated artifacts are git-ignored and MUST be rebuilt on deploy
The SQLite databases and the FAISS schema index are **not** in the repo (see `.gitignore`):
- `db/*.db` — created by `python db/setup_sqlite.py`
- `schema_index/` — created by `python main.py --build-index`

**The deploy build step MUST run both commands**, in this order, or the app will start
but every query will fail:
```bash
python db/setup_sqlite.py
python main.py --build-index
```

### 2b. `server.py` is DEV-ONLY — do not use it in production
`server.py` hardcodes `port=8000` and `reload=True`. DigitalOcean App Platform injects a
`PORT` environment variable (commonly 8080) and expects the app to bind `0.0.0.0:$PORT`.
**Production run command must be uvicorn directly, no reload:**
```bash
uvicorn api.app:app --host 0.0.0.0 --port $PORT
```
(App Platform sets `$PORT`. Do NOT run `python server.py` in production.)

### 2c. CORS currently allows ONLY localhost — must add the deployed frontend URL
`api/app.py` has:
```python
allow_origins=["http://localhost:5173", "http://localhost:4173"]
```
After the frontend is deployed and you know its URL (e.g. `https://sql-agents-frontend-xxxxx.ondigitalocean.app`),
**add that exact origin** to `allow_origins`, commit, and push. Without this, the browser
will block the frontend's requests with a CORS error. Prefer reading it from an env var:
```python
import os
_frontend = os.getenv("FRONTEND_ORIGIN", "")
allow_origins=[o for o in ["http://localhost:5173", "http://localhost:4173", _frontend] if o]
```

### 2d. Secrets — never commit them
`GROQ_API_KEY` must be set as an **encrypted environment variable** in the DigitalOcean
App settings, NOT in the repo. The `.env` file is git-ignored and stays local.

### 2e. Redis is optional but matters for scaling
`config.py` defaults `REDIS_URL` to `redis://localhost:6379/0`. There is NO Redis on a
fresh App Platform service, so the code falls back to the in-memory `_InMemoryRedis`.
- **Single instance (1 container):** in-memory fallback works fine. Fine for a first deploy.
- **Multiple instances/replicas:** in-memory state is NOT shared between containers, so the
  Blackboard deduplication breaks. If you scale beyond 1 instance, provision a
  **DigitalOcean Managed Redis** (or Valkey) database and set `REDIS_URL` to its connection
  string. Keep instance count = 1 until then.

### 2f. SSE needs buffering disabled (already handled)
The routes already send `X-Accel-Buffering: no` and `Cache-Control: no-cache`. App Platform
generally streams fine. If responses appear to "hang then dump all at once," revisit this.

### 2g. Model download at build time
`sentence-transformers` downloads the `all-MiniLM-L6-v2` model (~90 MB) the first time
`--build-index` runs. This happens during the build and needs internet (App Platform build
has it). It adds time to the first build and increases image size — expected.

---

## 3. Environment variables to set in DigitalOcean

| Variable | Required | Value / Notes |
|---|---|---|
| `GROQ_API_KEY` | **Yes** | The Groq key. Mark as **encrypted/secret**. Lives only here, never in git. |
| `GROQ_MODEL` | No | Defaults to `llama-3.3-70b-versatile`. |
| `FRONTEND_ORIGIN` | Recommended | The deployed frontend URL, for CORS (see 2c). |
| `REDIS_URL` | No | Only if using Managed Redis (see 2e). |
| `SQLITE_DB_PATH` | No | Defaults to `./db`. Leave default. |
| `SCHEMA_INDEX_PATH` | No | Defaults to `./schema_index`. Leave default. |
| `PORT` | Auto | Injected by App Platform. Do not set manually. |
| `PYTHONIOENCODING` | Recommended | Set to `utf-8` to avoid encoding errors. |

---

## 4. Recommended path — DigitalOcean App Platform (PaaS)

This is the simplest path and pairs with the existing GitHub Actions CI (CI checks the code;
App Platform deploys it automatically after you push).

### Step-by-step
1. **Create the app**: DigitalOcean dashboard → **Apps** → **Create App** → connect GitHub →
   pick `arshmankhawar/sql-agents-backend`, branch `main`, **Autodeploy on push** = ON.
2. **Resource type**: App Platform should detect Python. Set it as a **Web Service**.
3. **Build command** (Settings → Components → your service → Build Command):
   ```bash
   pip install -r requirements.txt && python db/setup_sqlite.py && python main.py --build-index
   ```
4. **Run command**:
   ```bash
   uvicorn api.app:app --host 0.0.0.0 --port $PORT
   ```
5. **HTTP port**: set to `$PORT` / 8080 (App Platform usually auto-detects).
6. **Environment variables**: add `GROQ_API_KEY` (encrypted), `PYTHONIOENCODING=utf-8`, and
   later `FRONTEND_ORIGIN`.
7. **Instance size**: smallest (Basic, ~$5/mo) is enough to start. **Instance count = 1**
   (see Redis note 2e).
8. **Deploy** and watch the build logs. The build runs the SQLite + FAISS index steps.
9. **Get the public URL** (e.g. `https://sql-agents-backend-xxxxx.ondigitalocean.app`).

### Optional: define infra as code
You can commit a `.do/app.yaml` (App Spec) so the configuration is version-controlled.
Ask the user if they want this; otherwise the dashboard UI is fine for a first deploy.

---

## 5. Alternative path — Droplet + Docker (more control, more work)

Only if the user explicitly wants a raw server. Outline:
1. Create an Ubuntu Droplet.
2. Install Docker (or Python directly).
3. Write a `Dockerfile`: install deps → run `setup_sqlite.py` + `--build-index` → CMD uvicorn.
4. Run a managed/containerized Redis if scaling.
5. Put nginx in front for TLS, and **disable proxy buffering** for the SSE routes
   (`proxy_buffering off;`) or streaming breaks.
6. Set up a process manager (systemd / Docker restart policy) to keep it running.

There is no Dockerfile in the repo yet — create one if going this route.

---

## 6. Post-deploy verification

```bash
# 1. Health check (adjust path to the actual health route under /api/v1)
curl https://<backend-url>/api/v1/health

# 2. SSE smoke test — should stream events, not hang
curl -N -X POST https://<backend-url>/api/v1/compare \
  -H "Content-Type: application/json" \
  -d '{"query":"Compare average salaries between tech_startup and airport","mode":1}'
```
Then open the deployed frontend in a browser, submit a query, and confirm the live task
feed + final answer render with no CORS errors in the browser console.

---

## 7. The full CI/CD picture once deployed

```
push to main → GitHub Actions CI (lint, import check, unit tests) → ✅
            → DigitalOcean App Platform auto-pulls → build (deps + DB + index)
            → run (uvicorn) → live
```
CI is the quality gate; App Platform autodeploy is the CD half.

---

## 8. Quick reference — local run (for parity / debugging)

```bash
python db/setup_sqlite.py        # one-time: build SQLite DBs
python main.py --build-index     # one-time: build FAISS index
python server.py                 # dev server on :8000 (reload on)
```
On Windows prefix with `PYTHONIOENCODING=utf-8`. See `README.md` and
`docs/ARCHITECTURE_DIAGRAM.md` for the full architecture
(planning → DAG execution → Blackboard → synthesis).
