"""
config.py — Centralized configuration for the Virus Dataset AI Agent.
Import in any module with: from config import *  or  from config import MCP_SERVER_URL, ...

Secrets/credentials are NOT stored here — the app's live in a single,
gitignored .env file:
  - .env.app  (loaded by backend/main.py) → ALBERT_API_KEY, JWT_SECRET, ...
See .env.app.example for the expected keys. The MCP data server is a separate
service (its own repo, viromeatlas_mcp) with its own .env and S3 credentials;
this app reaches it over HTTP via MCP_SERVER_URL below.

A user account is required to use the app — there is no guest mode. Accounts
live in the SQLite database (DB_PATH), managed by the FastAPI backend
(backend/auth.py): bcrypt password hashes + stateless JWT sessions.
"""

import os

# ==================== PATHS ==================== #
LOG_DIR      = "logs"

APP_ENV_PATH = ".env.app"

# Single SQLite database holding users (email, bcrypt hash, role), their
# conversations and messages, plus a little app_meta (schema version, etc.).
# Kept in its own subdirectory so it can be given a persistent Docker volume
# (see docker-compose.yml) without bind-mounting the whole app directory.
DB_PATH = os.path.join("auth_data", "viromechat.db")


# ==================== ROLES & AUTH (React/FastAPI API) ==================== #
# Three roles, ordered by privilege, enforced by the FastAPI backend (backend/).
# 'dev' sits between user and admin: full chat + expert mode + MCP/log
# introspection, but no user administration.
VALID_ROLES = ("user", "dev", "admin")
ROLE_LEVEL = {"user": 0, "dev": 1, "admin": 2}

# JWT signing config for the API. JWT_SECRET MUST be set to a long random value
# in production (env / .env.app) — the default here is only a dev convenience
# and is intentionally obvious so a misconfigured prod deployment stands out.
JWT_SECRET      = os.environ.get("JWT_SECRET", "dev-insecure-change-me")
JWT_ALGORITHM   = "HS256"
JWT_EXPIRE_MIN  = int(os.environ.get("JWT_EXPIRE_MIN", "720"))  # 12h


def _admin_emails() -> set[str]:
    """Emails granted the 'admin' role, from the ADMIN_EMAILS secret/env
    (comma-separated). Read lazily so tests and the app can set it via env.
    Matching is case-insensitive — emails double as usernames and are stored
    lower-cased."""
    raw = os.environ.get("ADMIN_EMAILS", "")
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def load_env_file(env_path: str) -> None:
    """
    Load KEY=VALUE pairs from a .env-style file into os.environ, without
    overriding variables already set in the real environment (so a real
    deployment's env vars always win over a local .env file).
    """
    if not os.path.exists(env_path):
        return
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip()

# ── Albert API Configuration ──────────────────────────────────────────────────
ALBERT_BASE_URL      = "https://albert.api.etalab.gouv.fr/v1"
ALBERT_TIMEOUT       = 120          # seconds — large models can be slow
ALBERT_MODEL_DEFAULT = "openai/gpt-oss-120b"  # fallback if model list fails

# The free Albert API is heavily rate-limited (HTTP 429), especially on the
# large models — retry a few times, honoring the server's Retry-After header
# when present and otherwise backing off exponentially up to this cap (seconds).
ALBERT_MAX_RETRIES       = 5
ALBERT_RETRY_BACKOFF_CAP = 30

# ==================== MCP ==================== #
# URL of the external MCP data server (separate repo: viromeatlas_mcp). The
# default assumes it runs on the same host on :8000; override via env var for a
# containerized or remote deployment. In Docker, docker-compose.yml sets this to
# http://host.docker.internal:8000/mcp so the api container can reach an MCP
# running on the host.
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8000/mcp")

# ==================== AGENT DEFAULTS ==================== #
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9

DEFAULT_PRESENCE_PENALTY = -0.2
DEFAULT_FREQUENCY_PENALTY = 0.2

DEFAULT_SEED = 42

DEFAULT_MAX_COMPLETION_TOKENS = 4096
DEFAULT_PARALLEL_TOOL_CALLS = False

DEFAULT_MAX_TOOL_CALLS = 7
DEFAULT_MAX_TOOL_CONTENT = 6000

# How many user questions the model keeps context for before the
# conversation memory resets. Tool calls/results are stripped from history
# after each turn (see clean_history_messages in backend/albert.py), so this
# only bounds the number of user/assistant text exchanges kept.
MAX_CONTEXT_TURNS = 5

# ==================== UI DEFAULTS ==================== #
DEFAULT_PREVIEW_ROWS   = 50
DEFAULT_WIKIPEDIA_LIMIT = 4000