# 🦠 Viromech@t: Virus Dataset AI Agent

[![CI](https://github.com/Romumrn/viromechat/actions/workflows/ci.yml/badge.svg)](https://github.com/Romumrn/viromechat/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Latest release](https://img.shields.io/github/v/tag/Romumrn/viromechat?label=release)](https://github.com/Romumrn/viromechat/tags)

By October 2025, ChatGPT was serving more than 800 million weekly active users, and the count
kept climbing past 900 million by early 2026 ([OpenAI figures, reported by TechCrunch](https://techcrunch.com/2026/02/27/chatgpt-reaches-900m-weekly-active-users)).
Researchers are part of that wave. They ask chatbots to summarize a family, to count species, to
place a sample on a map. The catch is well known: a general-purpose model will happily invent a
species count or a coordinate when it does not actually know, and in virology a confident wrong
answer is worse than no answer.

Viromech@t is built on the opposite principle. The language model never sees the raw data. It acts
only through a small set of audited tools served by a separate MCP server
([`viromeatlas_mcp`](https://github.com/Romumrn/viromeatlas_mcp)), so every number, chart, and
citation it returns comes from a real query rather than a plausible guess. It reads like a chatbot
and answers like a database.

> **Architecture note.** Viromech@t runs as a **React front-end + FastAPI backend** (this document).
> The API and the role model are covered in [README_API.md](README_API.md). The tool and resource
> reference lives with the server itself, in [`viromeatlas_mcp`](https://github.com/Romumrn/viromeatlas_mcp).

## Project Context

This project is developed within the framework of **SHAPE-Med@Lyon** and contributes to the
structuring research initiative [**Virome@tlas**](https://www.shape-med-lyon.fr/projets/structurants-vague-1/virometlas/).

*Virome@tlas* aims to build an integrated digital platform for large-scale exploration and
surveillance of the global virosphere. It uses publicly available sequencing data to study virus
diversity, virus–host interactions, and how viruses are distributed across ecosystems, all within a
transdisciplinary **One Health** framework that spans human, animal, and environmental health.

Viromech@t is the conversational front door to that effort. You ask questions in plain language; it
runs the queries, draws the charts, pulls the references, and keeps every biological claim tied to
something a tool actually returned.

## Architecture

Three processes, one job each. The browser only ever talks to the **FastAPI backend**. That backend
asks the **Albert API** what to do next, calls the **MCP server** to actually do it, then feeds the
result back to Albert for the following step. The loop continues until Albert stops asking for tools
and writes its answer. Everything the user sees is streamed live over **Server-Sent Events**, in
order: status, tool calls, figures, sources, then the final answer.

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
│  :5173 (dev)     │                │   roles, SSE  :8080 │ resources│   FastMCP —               │
└──────────────────┘                └─────────────────────┘          │   data access & tools     │
                                       MCP_SERVER_URL ───────────────►│   :8000                   │
                                                                     └───────────────────────────┘
```

* **`frontend/`**: React + Vite + TypeScript + Tailwind SPA (hand-written shadcn-style UI in
  `src/components/ui.tsx`). Pages: Login, Register, Chat, Account, Admin, Dev. In production it is
  built (`frontend/dist`) and served by the backend as a single origin (no CORS).
* **`backend/`**: FastAPI app. Owns authentication (JWT), the three-role model, conversation
  persistence, and the agent loop (`backend/agent.py`, an async generator of SSE events). It never
  touches a dataframe or an S3 credential, it lists the MCP server's tools, forwards them to the
  [Albert API](https://albert.api.etalab.gouv.fr) (French government sovereign LLM infrastructure)
  for tool-calling, and dispatches each call back to the MCP server. It is deliberately generic:
  it reads each tool's JSON schema to decide which configured defaults apply, rather than
  hardcoding tool names. It reuses the repo-root modules unchanged: `db.py` (SQLite),
  `config.py`, `prompt.py`, `logging_utils.py`.
* **MCP server**: owns the datasets and every tool's business logic and guardrails. It lives in
  its **own repository** ([`viromeatlas_mcp`](https://github.com/Romumrn/viromeatlas_mcp)); this app
  only reaches it over HTTP via `MCP_SERVER_URL`, and never sees the data itself. It never talks to
  Albert directly.

The backend and the MCP server each read their own secrets file. The backend's is
`.env.app` (see [Configuration](#configuration)); the MCP server's `.env` lives in its own repo.
Albert needs only the API key in `.env.app`.


## Features

* Natural-language querying of viral taxonomy and virus–host relationships
* Authoritative taxonomy/acronym resolution via NCBI Taxonomy (e.g. `HIV` → `Lentivirus humimdef1`)
* SQL queries over a large virus–host dataset, run server-side through the MCP server's tools
* Interactive Plotly charts and geographic maps (scroll-to-zoom enabled)
* Wikipedia and PubMed lookups for biological/clinical background, with mandatory inline citations
* **Voice input**: dictate a question with the 🎙️ mic button; transcribed live, word-by-word, in the browser (Web Speech API, Chrome/Edge/Safari)
* Multi-conversation history (ChatGPT-style sidebar: new / switch / rename / delete), persisted per user in SQLite
* Sliding conversation memory over the last few Q&A turns (tool-call traces stripped after each turn)
* **Three roles**: `user` < `dev` < `admin`, with Expert mode, a Dev console, and an Admin console (see [Roles](#roles-accounts))
* PMID hallucination guard: any PMID not returned by an actual `pubmed_search` call is stripped from the answer
* Per-tool-call live status line (search keyword only, no clutter for dataset/map calls) with full detail logged to disk
* In-app 🚩 **"Report an error"** button (question, answer, executed code, and recent logs bundled into a report file)


## Scientific Guardrails

Enforced through the system prompt, tool-level validation, and post-processing on the backend:

* No invention of taxa, species counts, coordinates, or any biological fact. Every statement traces back to a tool call.
* Acronyms (HIV, MPOX, SARS, and so on) must be resolved via `ncbi_taxonomy_search` before being used in any other tool.
* `query_host_sql` only allows read-only `SELECT` statements and rejects bare `SELECT *`.
* `create_map` rejects any map that leaves the sample identifier (`primary_id`) out of its hover data. Every plotted point stays traceable to its exact BioSample sample.
* **PMID hallucination guard**: a whitelist of real PMIDs is built from actual `pubmed_search` calls in the conversation; any PMID outside that whitelist is stripped from the final answer and logged.
* Bracket-style citation artifacts (e.g. `【4†L13-L17】`, a known gpt-oss-120b browsing-tool artifact) are stripped. Citations must be real Markdown links to a URL actually returned by a tool.
* If information is absent from the datasets and tools, the agent must say so explicitly rather than guess.

## Example Queries

* "Give me information about Orthopoxvirus, is it a genus or a family, and how many species does it include?"
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
`_clean_history_messages` / `build_context_window` in `backend/albert.py`), only the
user/assistant text is kept. This still lets the model resolve follow-ups like *"and which family
is that genus part of?"* without the subject being restated, and without replaying every past
tool call/result to Albert on every new question.

Memory is a **sliding window**: the last `MAX_CONTEXT_TURNS` question/answer exchanges (`config.py`,
default **5**, adjustable at runtime in Expert mode) are sent to Albert. Older turns simply fall
out of the prompt but stay on screen and in the database, "unbounded scrollback, bounded prompt",
like ChatGPT. A failed turn (API timeout/error) is never added to memory. The full conversation is
saved to SQLite per user, so it survives page reloads and new sessions.

## Roles & Accounts

A user account is required. There is no guest mode. Accounts are fully local (no external
identity provider, no email service): the backend issues a **JWT** on login and stores users in
the SQLite database (`auth_data/viromechat.db`), with bcrypt-hashed passwords.

Three roles, in ascending order of privilege: `user` < `dev` < `admin`:

* **user**: chat, manage their own conversations, change their password.
* **dev**: the above **+ Expert mode** (model, sampling parameters, agent limits), the **MCP tool
  tester**, and the **agent log viewer** (Dev page).
* **admin**: the above **+ the Administration page**: list/search users, change roles, delete
  users, platform stats, and read any user's conversations.

Bootstrap the first admin with `ADMIN_EMAILS` in `.env.app` (that email gets `admin` on
registration); admins then promote others from the UI.

* **Registration**: first name, last name, institutional email (the email doubles as the
  username), and a password that must be **at least 12 characters with 1 lowercase, 1 uppercase,
  1 digit and 1 special character** (a live checklist shows each rule as you type). The email
  domain must **not** be a free webmail provider (Gmail, Outlook, Yahoo, iCloud, Proton, Orange,
  and others, see `_BLOCKED_EMAIL_DOMAINS` in `backend/auth.py`). This is a quick blocklist, not a
  real institution allowlist. Someone with their own custom domain still gets through; the point is
  just to steer people toward their work address.
* **Optional invite code**: set `REGISTRATION_CODE` in `.env.app` to gate registration behind a
  shared code. Leave it unset to keep registration open. The expected value is only ever read from
  the secret, never stored in the source.
* **Login**: email + password, returns a JWT kept in the browser (`localStorage`) and sent as a
  `Bearer` header. Token lifetime is `JWT_EXPIRE_MIN` (default 720 min = 12 h).


## Datasets

The datasets are owned and served by the MCP server, the app never stores or loads them itself, it
only queries them through the server's tools. For the datasets, their schemas and how they are
stored, see the MCP repo: [`viromeatlas_mcp`](https://github.com/Romumrn/viromeatlas_mcp).


## Setup

### Requirements

* Python 3.10+
* Node.js 18+ (for the React front-end)
* An **Albert API key** ([albert.api.etalab.gouv.fr](https://albert.api.etalab.gouv.fr))
* A running **MCP server** (separate repo: [`viromeatlas_mcp`](https://github.com/Romumrn/viromeatlas_mcp)), reachable at
  `MCP_SERVER_URL`, it owns the S3 dataset access

Python dependencies live under [`requirements/`](requirements/): `api.txt` is the self-contained
FastAPI backend list; `dev.txt` adds `pytest` for the [test suite](#testing). (The MCP server's
dependencies live in its own repo.)

### Configuration

This app has a single secrets file. The MCP server's S3 credentials live in *its* repo, not here.

* **`.env.app`**: read by the backend (copy from [`.env.app.example`](.env.app.example)):

  ```bash
  ALBERT_API_KEY=sk-...
  # Long random value in production, anyone who knows it can mint valid tokens:
  #   python -c "import secrets; print(secrets.token_hex(32))"
  JWT_SECRET=...
  # Optional:
  # JWT_EXPIRE_MIN=720
  # ADMIN_EMAILS=alice@lab.fr,bob@univ.fr   # bootstrap the first admin(s)
  # REGISTRATION_CODE=...                   # shared invite code gate
  ```

  User accounts need no further configuration, the SQLite database
  (`auth_data/viromechat.db`) is created automatically on first run.

  Point the backend at the MCP server with `MCP_SERVER_URL` (env var; default
  `http://localhost:8000/mcp`). The S3 credentials themselves are configured in the MCP repo.

`.env.app` is gitignored. Non-secret configuration (model defaults, sampling parameters, timeouts,
the MCP server URL, and so on) lives in `config.py`.

### Running (development)

Three terminals:

```bash
# 1. MCP data server, from its OWN repo (viromeatlas_mcp); see that repo's README
cd ../viromeatlas_mcp && python3 server_mcp.py

# 2. FastAPI backend (needs .env.app); interactive docs at http://localhost:8080/docs
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

This repo's compose starts a single `api` container (FastAPI + built React SPA, one origin, no
CORS). The MCP server runs separately (its own repo / deployment); point `MCP_SERVER_URL` at it.
Set a real `JWT_SECRET` in `.env.app` first.

```bash
# MCP_SERVER_URL defaults to http://host.docker.internal:8000/mcp (an MCP on the host)
docker compose up --build            # api → http://localhost:8080
```

* **`docker/Dockerfile.api`** → `api` service (FastAPI + built SPA, port 8080), reaching the MCP
  server at `MCP_SERVER_URL`

Secrets are excluded from the image by `.dockerignore`, the backend loads only its own `.env.app`
at runtime. The SQLite database lives in a **host bind-mount**
(`./auth_data/viromechat.db`), readable/backup-able from the host. Because a bind-mount keeps the
host directory's ownership, which may not match the container's non-root `app` user (uid 1000) ,
the API container starts from an entrypoint (`docker/entrypoint-api.sh`) that briefly runs as root
to `chown` the mounted `auth_data/` and `logs/`, then drops privileges to run as `app`.


## Testing

Unit tests cover the pure helper logic (argument parsing, citation/PMID guardrails, SQL
validation, table/figure formatting, auth/password rules, and more), see `tests/`. They run in CI on
every push and pull request to `main` (see `.github/workflows/ci.yml`).

```bash
pip install -r requirements/api.txt -r requirements/dev.txt
pytest
```

## ⚠️ Disclaimer

This system is intended for exploratory and research support purposes only. All outputs should
be independently verified before use in scientific or medical contexts. Dataset coverage reflects
what has been sequenced and deposited in public repositories, it does not reflect epidemiological
prevalence or clinical severity.

## License

GNU General Public License v3.0 (GPL-3.0), see [LICENSE](LICENSE).
