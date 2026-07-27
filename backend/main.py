"""
backend/main.py — FastAPI application entry point.

Run locally:  uvicorn backend.main:app --reload --port 8080
(from the repo root, so `import db`, `import config`, ... resolve).

Assembles the routers, initializes the SQLite DB (running the idempotent legacy
migration), enables CORS for the Vite dev server, and — in production — serves
the built React SPA from frontend/dist so the whole product is one origin.
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import db
from config import APP_ENV_PATH, load_env_file

# ALBERT_API_KEY / JWT_SECRET / ADMIN_EMAILS / REGISTRATION_CODE come from
# .env.app (or the real environment, which always wins — see load_env_file).
load_env_file(APP_ENV_PATH)

from backend.routers import (  # noqa: E402  (import after env is loaded)
    auth_routes, chat_routes, conversation_routes, admin_routes, dev_routes,
)

app = FastAPI(title="Viromech@t API", version="1.0.0")

# CORS: only needed when the front runs on a different origin (Vite dev on
# :5173). In production the SPA is same-origin (served below) so this is inert.
_origins = os.environ.get(
    "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    db.init_db()


@app.get("/api/health")
def health():
    return {"status": "ok"}


for r in (auth_routes.router, chat_routes.router, conversation_routes.router,
          admin_routes.router, dev_routes.router):
    app.include_router(r)


# ── Serve the built React SPA (production) ───────────────────────────────────
# When frontend/dist exists, mount it and fall back to index.html for client-
# side routes. In dev this directory is absent and Vite serves the front.
_DIST = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(_DIST, "assets")), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        # API 404s already returned above; anything else serves the SPA shell.
        candidate = os.path.join(_DIST, full_path)
        if full_path and os.path.isfile(candidate):
            return FileResponse(candidate)
        return FileResponse(os.path.join(_DIST, "index.html"))
