#!/bin/sh
# Entrypoint for the API container. Runs as root only long enough to fix the
# ownership of the host bind-mounts (./auth_data holding the SQLite DB, ./logs),
# then drops privileges and runs the FastAPI app as the non-root app user.
#
# A single uvicorn worker on purpose: db.py shares one process-wide SQLite
# connection, and the SSE chat stream keeps that state in-process.
set -e

chown -R app:app /app/auth_data /app/logs 2>/dev/null || true

exec gosu app uvicorn backend.main:app --host 0.0.0.0 --port 8080
