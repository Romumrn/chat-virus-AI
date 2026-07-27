"""FastAPI backend for Viromech@t — the REST + SSE API that replaces the
Streamlit monolith (app.py). Reuses the root modules db.py, config.py,
prompt.py and logging_utils.py unchanged; the agent loop and the ALBERT/MCP
helpers were extracted from app.py into backend.agent / backend.albert.
"""
