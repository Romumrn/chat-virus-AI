# 🦠 Viromech@t — Virus Dataset AI Agent

[![CI](https://github.com/Romumrn/viromechat/actions/workflows/ci.yml/badge.svg)](https://github.com/Romumrn/viromechat/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Latest release](https://img.shields.io/github/v/tag/Romumrn/viromechat?label=release)](https://github.com/Romumrn/viromechat/tags)

A conversational agent for exploring viral taxonomy and virus–host data, built on a strict
tool-calling architecture: the LLM never sees raw data directly, it can only act through a
small set of audited tools exposed by a separate [MCP server](README_MCP.md).

> **Architecture note** — Viromech@t runs as a **React front-end + FastAPI backend**
> (this document). See [README_API.md](README_API.md) for the API/roles deep-dive and
> [README_MCP.md](README_MCP.md) for the tool/resource reference.

## Project Context

This project is developed within the framework of **SHAPE-Med@Lyon** and contributes to the
structuring research initiative [**Virome@tlas**](https://www.shape-med-lyon.fr/projets/structurants-vague-1/virometlas/).

*Virome@tlas* aims to build an integrated digital platform for large-scale exploration and
surveillance of the global virosphere, leveraging publicly available sequencing data to analyze
virus diversity, virus–host interactions, and ecological distribution patterns within a
transdisciplinary **One Health** framework spanning human, animal, and environmental health.

Viromech@t supports this effort as a research companion tool combining deterministic dataset
querying, transparent visualization, controlled external knowledge retrieval, and strict
grounding of every biological statement in tool output.

## Architecture

Three independent processes. The browser talks only to the **FastAPI backend**; the backend
round-trips to the **Albert API** to decide what to do, then to the **MCP server** to actually
do it, feeding each result back into the next round-trip to Albert — repeating until Albert
returns a final answer instead of more tool calls. The agent turn is streamed to the browser
over **Server-Sent Events** (status → tool calls → figures → sources → final answer).

```
                         ┌─────────────────────────┐
                         │        Albert API       │
                         │  (sovereign LLM, OpenAI- │
                         │   compatible tool-calls) │
                         └────────────┬────────────┘
                            history + │ tool_calls /
                             tool specs│ final answer
                                      ▼
┌──────────────────┐   REST + SSE   ┌─────────────────────┐   HTTP   ┌───────────────────────────┐
│  React front     │ ─────────────► │   FastAPI backend   │ ───────► │   MCP server              │
│  (Vite + TS +    │ ◄───────────── │   (backend/)        │ ◄─────── │   (separate repo:         │
│   Tailwind)      │   JWT auth     │   agent loop, auth, │  tools + │    viromeatlas_mcp)       │
│  :5173 (dev)     │                │   roles, SSE  :8080 │ resources│   FastMCP — taxonomy CSV  │
└──────────────────┘                └─────────────────────┘          │   + S3 host via DuckDB     │
                                       MCP_SERVER_URL ───────────────►│   :8000                   │
                                                                     └───────────────────────────┘
```

* **`frontend/`** — React + Vite + TypeScript + Tailwind SPA (hand-written shadcn-style UI in
  `src/components/ui.tsx`). Pages: Login, Register, Chat, Account, Admin, Dev. In production it is
  built (`frontend/dist`) and served by the backend as a single origin (no CORS).
* **`backend/`** — FastAPI app. Owns authentication (JWT), the three-role model, conversation
  persistence, and the agent loop (`backend/agent.py`, an async generator of SSE events). It never
  touches a dataframe or an S3 credential — it lists the MCP server's tools, forwards them to the
  [Albert API](https://albert.api.etalab.gouv.fr) (French government sovereign LLM infrastructure)
  for tool-calling, and dispatches each call back to the MCP server. It is deliberately generic:
  it reads each tool's JSON schema to decide which configured defaults apply, rather than
  hardcoding tool names. It reuses the repo-root modules unchanged: `db.py` (SQLite),
  `config.py`, `prompt.py`, `logging_utils.py`.
* **MCP server** — owns the datasets, the DuckDB/S3 connection, and every tool's business
  logic and guardrails. It now lives in its **own repository**
  ([`viromeatlas_mcp`](../viromeatlas_mcp)); this app only reaches it over HTTP via
  `MCP_SERVER_URL`. It never talks to Albert directly.

The backend and the MCP server each read their own separate secrets file — the backend's is
`.env.app` (see [Configuration](#configuration)); the MCP server's `.env` lives in its own repo.
Albert needs only the API key in `.env.app`.


## Features

* Natural-language querying of viral taxonomy and virus–host relationships
* Authoritative taxonomy/acronym resolution via NCBI Taxonomy (e.g. `HIV` → `Lentivirus humimdef1`)
* SQL queries against a multi-GB virus–host Parquet dataset on S3, without ever loading it into memory (DuckDB + `httpfs`/`spatial`)
* Interactive Plotly charts and geographic maps (scroll-to-zoom enabled)
* Wikipedia and PubMed lookups for biological/clinical background, with mandatory inline citations
* **Voice input** — dictate a question with the 🎙️ mic button; transcribed live, word-by-word, in the browser (Web Speech API — Chrome/Edge/Safari)
* Multi-conversation history (ChatGPT-style sidebar: new / switch / rename / delete), persisted per user in SQLite
* Sliding conversation memory over the last few Q&A turns (tool-call traces stripped after each turn)
* **Three roles** — `user` < `dev` < `admin` — with Expert mode, a Dev console, and an Admin console (see [Roles](#roles-accounts))
* PMID hallucination guard: any PMID not returned by an actual `pubmed_search` call is stripped from the answer
* Per-tool-call live status line (search keyword only — no clutter for dataset/map calls) with full detail logged to disk
* In-app 🚩 **"Report an error"** button (question, answer, executed code, and recent logs bundled into a report file)


## Scientific Guardrails

Enforced through the system prompt, tool-level validation, and post-processing on the backend:

* No invention of taxa, species counts, coordinates, or any biological fact — every statement must trace back to a tool call.
* Acronyms (HIV, MPOX, SARS, …) must be resolved via `ncbi_taxonomy_search` before being used in any other tool.
* `query_host_sql` rejects bare `SELECT *` (it would pull ~65 columns, including a heavy geometry blob, across the whole S3 dataset) and only allows read-only `SELECT` statements.
* `create_map` rejects any map that doesn't include the sample identifier (`primary_id`) in its hover data — every plotted point must be traceable back to its exact BioSample sample.
* **PMID hallucination guard**: a whitelist of real PMIDs is built from actual `pubmed_search` calls in the conversation; any PMID outside that whitelist is stripped from the final answer and logged.
* Bracket-style citation artifacts (e.g. `【4†L13-L17】`, a known gpt-oss-120b browsing-tool artifact) are stripped — citations must be real Markdown links to a URL actually returned by a tool.
* If information is absent from the datasets and tools, the agent must say so explicitly rather than guess.

## Example Queries

* "Give me information about Orthopoxvirus — is it a genus or a family, and how many species does it include?"
* "Show a pie chart of genus distribution within Poxviridae."
* "World distribution of Poxviridae."
* "Tell me more about Polyomavirus infection pathway."
* "What is HBV, exactly, taxonomically?"


## Transparency & Logging

* Executed SQL/pandas code is shown in the "📚 Sources" panel of each response, next to the
  Wikipedia / PubMed / NCBI Taxonomy links used to build the answer.
* Every tool call is numbered and fully traced in `logs/agent_YYYY-MM.log`, including a preview of the actual response content (not just success/failure).
* Error reports submitted via the in-app "🚩 Report an error" button are saved to `logs/error_reports/`, bundled with the question, answer, executed code, and recent log lines. `dev`+ users can also tail the agent log from the Dev page.


## Conversation Memory

Each new question is answered with the previous questions and the model's final text answers as
context. Tool calls and their raw results are dropped from history right after each turn (see
`_clean_history_messages` / `build_context_window` in `backend/albert.py`) — only the
user/assistant text is kept. This still lets the model resolve follow-ups like *"and which family
is that genus part of?"* without the subject being restated, and without replaying every past
tool call/result to Albert on every new question.

Memory is a **sliding window**: the last `MAX_CONTEXT_TURNS` question/answer exchanges (`config.py`,
default **5**, adjustable at runtime in Expert mode) are sent to Albert. Older turns simply fall
out of the prompt but stay on screen and in the database — "unbounded scrollback, bounded prompt",
like ChatGPT. A failed turn (API timeout/error) is never added to memory. The full conversation is
saved to SQLite per user, so it survives page reloads and new sessions.

## Roles & Accounts

A user account is required — there is no guest mode. Accounts are fully local (no external
identity provider, no email service): the backend issues a **JWT** on login and stores users in
the SQLite database (`auth_data/viromechat.db`), with bcrypt-hashed passwords.

Three roles, ascending privilege — `user` < `dev` < `admin`:

* **user** — chat, manage their own conversations, change their password.
* **dev** — the above **+ Expert mode** (model, sampling parameters, agent limits), the **MCP tool
  tester**, and the **agent log viewer** (Dev page).
* **admin** — the above **+ the Administration page**: list/search users, change roles, delete
  users, platform stats, and read any user's conversations.

Bootstrap the first admin with `ADMIN_EMAILS` in `.env.app` (that email gets `admin` on
registration); admins then promote others from the UI.

* **Registration** — first name, last name, institutional email (the email doubles as the
  username), and a password that must be **at least 12 characters with 1 lowercase, 1 uppercase,
  1 digit and 1 special character** (a live checklist shows each rule as you type). The email
  domain must **not** be a free webmail provider (Gmail, Outlook, Yahoo, iCloud, Proton, Orange,
  … — see `_BLOCKED_EMAIL_DOMAINS` in `backend/auth.py`). This is a quick blocklist, not a real
  institution allowlist — someone with their own custom domain still gets through; the goal is
  just to steer people to their work address.
* **Optional invite code** — set `REGISTRATION_CODE` in `.env.app` to gate registration behind a
  shared code. Leave it unset to keep registration open. The expected value is only ever read from
  the secret, never stored in the source.
* **Login** — email + password, returns a JWT kept in the browser (`localStorage`) and sent as a
  `Bearer` header. Token lifetime is `JWT_EXPIRE_MIN` (default 720 min = 12 h).


## Datasets

| Dataset | Storage | Description |
|---|---|---|
| **Taxonomy** (`df_taxo`) | Local CSV (`data/TAXONOMY.csv`), loaded fully into memory | NCBI Taxonomy, enriched with genome assembly availability, SRA sequencing activity, and GBIF biodiversity observations. One row per taxon. |
| **Virus–host occurrences** (`host` / `df_host`) | Parquet on S3, queried on demand via DuckDB — never fully loaded | SRA/GenBank/BioSample samples linked to host & virus taxonomy, geographic location (as a `GEOMETRY` point column), and disease status. |

Column-by-column descriptions of both datasets are **not hardcoded in the client** — they are
published by the MCP server as resources (`resource://datasets/taxonomy/schema` and
`resource://datasets/host/schema`) and read once per conversation by the backend, which folds them
into the system prompt. This means the two datasets' schemas can change server-side without any
client code change.


## Setup

### Requirements

* Python 3.10+
* Node.js 18+ (for the React front-end)
* An **Albert API key** ([albert.api.etalab.gouv.fr](https://albert.api.etalab.gouv.fr))
* A running **MCP server** (separate repo: [`viromeatlas_mcp`](../viromeatlas_mcp)), reachable at
  `MCP_SERVER_URL` — it owns the S3 dataset access

Python dependencies live under [`requirements/`](requirements/): `api.txt` is the self-contained
FastAPI backend list; `dev.txt` adds `pytest` for the [test suite](#testing). (The MCP server's
dependencies live in its own repo.)

### Configuration

This app has a single secrets file. The MCP server's S3 credentials live in *its* repo, not here.

* **`.env.app`** — read by the backend (copy from [`.env.app.example`](.env.app.example)):

  ```bash
  ALBERT_API_KEY=sk-...
  # Long random value in production — anyone who knows it can mint valid tokens:
  #   python -c "import secrets; print(secrets.token_hex(32))"
  JWT_SECRET=...
  # Optional:
  # JWT_EXPIRE_MIN=720
  # ADMIN_EMAILS=alice@lab.fr,bob@univ.fr   # bootstrap the first admin(s)
  # REGISTRATION_CODE=...                   # shared invite code gate
  ```

  User accounts need no further configuration — the SQLite database
  (`auth_data/viromechat.db`) is created automatically on first run.

  Point the backend at the MCP server with `MCP_SERVER_URL` (env var; default
  `http://localhost:8000/mcp`). The S3 credentials themselves are configured in the MCP repo.

`.env.app` is gitignored. Non-secret configuration (model defaults, sampling parameters, timeouts,
the MCP server URL, …) lives in `config.py`.

### Running (development)

Three terminals:

```bash
# 1. MCP data server — from its OWN repo (viromeatlas_mcp); see that repo's README
cd ../viromeatlas_mcp && python3 server_mcp.py

# 2. FastAPI backend (needs .env.app) — interactive docs at http://localhost:8080/docs
pip install -r requirements/api.txt
uvicorn backend.main:app --reload --port 8080

# 3. React front-end (Vite dev server, proxies /api → :8080)
cd frontend
npm install
npm run dev            # http://localhost:5173
```

If the backend runs on a non-default port, start Vite with
`API_PROXY_TARGET=http://localhost:<port> npm run dev`.

### Running with Docker

This repo's compose starts a single `api` container (FastAPI + built React SPA — one origin, no
CORS). The MCP server runs separately (its own repo / deployment); point `MCP_SERVER_URL` at it.
Set a real `JWT_SECRET` in `.env.app` first.

```bash
# MCP_SERVER_URL defaults to http://host.docker.internal:8000/mcp (an MCP on the host)
docker compose up --build            # api → http://localhost:8080
```

* **`docker/Dockerfile.api`** → `api` service (FastAPI + built SPA, port 8080), reaching the MCP
  server at `MCP_SERVER_URL`

Secrets are excluded from the image by `.dockerignore` — the backend loads only its own `.env.app`
at runtime. The SQLite database lives in a **host bind-mount**
(`./auth_data/viromechat.db`), readable/backup-able from the host. Because a bind-mount keeps the
host directory's ownership — which may not match the container's non-root `app` user (uid 1000) —
the API container starts from an entrypoint (`docker/entrypoint-api.sh`) that briefly runs as root
to `chown` the mounted `auth_data/` and `logs/`, then drops privileges to run as `app`.


## Testing

Unit tests cover the pure helper logic (argument parsing, citation/PMID guardrails, SQL
validation, table/figure formatting, auth/password rules, …) — see `tests/`. They run in CI on
every push and pull request to `main` (see `.github/workflows/ci.yml`).

```bash
pip install -r requirements/api.txt -r requirements/dev.txt
pytest
```

## ⚠️ Disclaimer

This system is intended for exploratory and research support purposes only. All outputs should
be independently verified before use in scientific or medical contexts. Dataset coverage reflects
what has been sequenced and deposited in public repositories — it does not reflect epidemiological
prevalence or clinical severity.

## License

GNU General Public License v3.0 (GPL-3.0) — see [LICENSE](LICENSE).
