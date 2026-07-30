# Viromech@t — React front-end + FastAPI backend

Three moving parts:

| Component | Path | Port | Role |
|-----------|------|------|------|
| **MCP server** | `server_mcp.py` | 8000 | Data & tools |
| **API** | `backend/` (FastAPI) | 8080 | REST + SSE, auth (JWT), roles |
| **Front** | `frontend/` (React + Vite + TS) | 5173 (dev) | Chat, Account, Admin, Dev UI |

The API reuses the root modules `db.py` (SQLite persistence), `config.py`,
`prompt.py`, `logging_utils.py`. The agent loop lives in `backend/agent.py` and
the ALBERT/MCP helpers in `backend/albert.py`.

## Roles

Three roles, ascending privilege — `user` < `dev` < `admin`:

- **user** — chat, manage their own conversations, change their password.
- **dev** — the above + Expert mode (model / temperature / top_p / max tool
  calls), MCP tool tester, and the agent log viewer.
- **admin** — the above + the Administration page: list/search users, change
  roles, delete users, platform stats, and read any user's conversations.

Bootstrap the first admin with `ADMIN_EMAILS` in `.env.app` (that email gets
`admin` on registration); admins then promote others from the UI. Passwords are
stored as bcrypt hashes and verified on login.

## Run in development

Three terminals from the repo root.

1. MCP data server (needs `.env.mcp` for S3 credentials):
```bash
python3 server_mcp.py
```

2. FastAPI backend (needs `.env.app` — `ALBERT_API_KEY`, `JWT_SECRET`, optional
`ADMIN_EMAILS` / `REGISTRATION_CODE`):
```bash
pip install -r requirements/api.txt
uvicorn backend.main:app --reload --port 8080
```
Interactive API docs at http://localhost:8080/docs.

3. React front-end (Vite dev server, proxies `/api` → `:8080`):
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:5173. If the backend runs on a non-default port, set
`API_PROXY_TARGET=http://localhost:<port>` before `npm run dev`.

## Run with Docker

Two containers (mcp + api); the API also serves the built React SPA, so it is a
single origin with no CORS. Set a real `JWT_SECRET` in `.env.app` first.
```bash
docker compose up --build          # starts mcp + api → http://localhost:8080
```
The SQLite database (`auth_data/viromechat.db`) lives in a host bind-mount, so
it survives rebuilds and is backup-able from the host.

## API surface

- `POST /api/auth/register`, `POST /api/auth/login`, `GET /api/auth/me`,
  `PUT /api/auth/me/password`
- `POST /api/chat` — **SSE stream** of the agent turn (status / tool_call /
  tool_result / figure / sources / final). Consumed by the front via
  `fetch()` + `ReadableStream` (EventSource can't send the Bearer header).
- `GET/POST/PATCH/DELETE /api/conversations[/{id}]`, `GET /api/conversations/{id}/messages`
- `POST /api/report` — submit a "Report an error" for an answer (stored in the DB for dev triage)
- `GET /api/admin/stats`, `GET /api/admin/users`, `PUT /api/admin/users/{email}/role`,
  `DELETE /api/admin/users/{email}`, `GET /api/admin/users/{email}/conversations`,
  `GET /api/admin/conversations/{id}/messages` (admin only)
- `GET /api/dev/mcp/tools`, `POST /api/dev/mcp/call`, `GET /api/dev/models`,
  `GET /api/dev/logs`, `GET /api/dev/reports`, `PATCH /api/dev/reports/{id}` (dev+ only)
