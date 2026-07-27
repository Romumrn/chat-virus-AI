"""backend/routers/dev_routes.py — DEV/ADMIN tooling: MCP tool introspection
and direct invocation, ALBERT model list (for the expert-mode picker), and a
tail of the agent log. Mounted at /api/dev; requires the 'dev' role or higher."""

import os

from fastapi import APIRouter, Depends, HTTPException, status
from fastmcp import Client

import db
from config import MCP_SERVER_URL, LOG_DIR
from backend import auth
from backend.albert import mcp_tools_to_openai_spec, unwrap_mcp_result, list_albert_models
from backend.schemas import McpToolCallIn

router = APIRouter(prefix="/api/dev", tags=["dev"])
_dev = auth.require_role("dev")


@router.get("/mcp/tools")
async def mcp_tools(_: dict = Depends(_dev)):
    """List the MCP server's tools with their OpenAI-format schemas — the same
    view the agent gets, useful for building/testing a call."""
    try:
        async with Client(MCP_SERVER_URL) as mcp:
            tools = await mcp.list_tools()
        return mcp_tools_to_openai_spec(tools)
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"MCP unreachable: {e}")


@router.post("/mcp/call")
async def mcp_call(body: McpToolCallIn, _: dict = Depends(_dev)):
    """Invoke an MCP tool directly and return its raw normalized result — a
    debugging shortcut so devs can exercise a tool without going through chat."""
    try:
        async with Client(MCP_SERVER_URL) as mcp:
            result = await mcp.call_tool(body.name, body.arguments or {})
        return unwrap_mcp_result(result)
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Tool call failed: {e}")


@router.get("/models")
def models(_: dict = Depends(_dev)):
    """Text-generation models available on ALBERT, for the expert-mode picker."""
    key = os.environ.get("ALBERT_API_KEY", "").strip()
    if not key:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "ALBERT_API_KEY not configured.")
    return {"models": list_albert_models(key)}


@router.get("/logs")
def logs(lines: int = 200, _: dict = Depends(_dev)):
    """Tail of the current month's agent log file (defaults to last 200 lines).
    The same log the Streamlit error-report feature attached."""
    from datetime import datetime
    lines = max(1, min(lines, 2000))
    path = os.path.join(LOG_DIR, f"agent_{datetime.now().strftime('%Y-%m')}.log")
    if not os.path.exists(path):
        return {"path": path, "lines": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            tail = [ln.rstrip() for ln in f.readlines()[-lines:]]
    except Exception as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Could not read log: {e}")
    return {"path": path, "lines": tail}
