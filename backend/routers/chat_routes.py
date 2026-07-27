"""backend/routers/chat_routes.py — the streaming chat endpoint.

POST /api/chat runs the agent loop and streams its events to the browser as
Server-Sent Events (text/event-stream). The React client reads the stream via
fetch()+ReadableStream (not EventSource, which can't send an auth header).

Event flow on the wire (each line: `data: <json>\n\n`):
  conversation → status/tool_call/tool_result/figure/sources/final/error → saved

Persistence happens inside the stream: the user turn is written immediately,
the assistant turn (with figures/sources/code) once the loop finishes.
"""

import json
import os

import plotly.io as pio
from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

import db
from config import (
    ALBERT_MODEL_DEFAULT, DEFAULT_TEMPERATURE, DEFAULT_TOP_P,
    DEFAULT_MAX_TOOL_CALLS, ROLE_LEVEL,
)
from backend import auth
from backend.agent import run_agent
from backend.albert import transcribe_audio
from backend.schemas import ChatIn

router = APIRouter(prefix="/api", tags=["chat"])


def _api_key() -> str:
    key = os.environ.get("ALBERT_API_KEY", "").strip()
    if not key:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "ALBERT_API_KEY not configured on the server.",
        )
    return key


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _title_from(text: str) -> str:
    t = " ".join((text or "").strip().split())
    return (t[:48] + "…") if len(t) > 48 else (t or "New conversation")


@router.post("/transcribe")
async def transcribe(file: UploadFile, user: dict = Depends(auth.get_current_user)):
    api_key = _api_key()
    audio_bytes = await file.read()
    text = transcribe_audio(audio_bytes, api_key)
    if not text:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Could not transcribe the recording — please try again or type your question.",
        )
    return {"text": text}


@router.post("/chat")
def chat(body: ChatIn, user: dict = Depends(auth.get_current_user)):
    api_key = _api_key()
    conn = db.get_conn()
    email = user["email"]
    query = (body.message or "").strip()
    if not query:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty message")

    # Resolve / create the conversation (ownership enforced for existing ones).
    if body.conversation_id is not None:
        conv = db.get_conversation(conn, body.conversation_id)
        if conv is None or conv["user_email"] != email:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")
        cid = body.conversation_id
        new_conv = False
    else:
        cid = db.create_conversation(conn, email, _title_from(query))
        new_conv = True

    # Expert-mode overrides are DEV+ only; plain users always get the defaults.
    is_expert = ROLE_LEVEL.get(user.get("role", "user"), 0) >= ROLE_LEVEL["dev"]
    model = body.model if (is_expert and body.model) else ALBERT_MODEL_DEFAULT
    temperature = body.temperature if (is_expert and body.temperature is not None) else DEFAULT_TEMPERATURE
    top_p = body.top_p if (is_expert and body.top_p is not None) else DEFAULT_TOP_P
    max_tool_calls = body.max_tool_calls if (is_expert and body.max_tool_calls) else DEFAULT_MAX_TOOL_CALLS

    # Bounded context replayed to ALBERT, built from stored history.
    from backend.albert import build_context_window
    from config import MAX_CONTEXT_TURNS
    history = build_context_window(db.list_messages_json(conn, cid), MAX_CONTEXT_TURNS)

    # Persist the user turn now so it survives even if the stream is dropped.
    db.add_message(conn, cid, "user", query)
    db.touch_conversation(conn, cid)

    async def event_stream():
        yield _sse({"type": "conversation", "id": cid,
                    "title": db.get_conversation(conn, cid)["title"], "new": new_conv})

        final_text = ""
        done = None
        try:
            async for ev in run_agent(
                model=model, api_key=api_key, user_query=query, username=email,
                temperature=temperature, top_p=top_p, max_tool_calls=max_tool_calls,
                history_messages=history,
            ):
                if ev["type"] == "done":
                    done = ev
                    continue
                if ev["type"] == "final":
                    final_text = ev["content"]
                if ev["type"] == "error":
                    final_text = ev["message"]
                yield _sse(ev)
        except Exception as e:  # defensive: never leave the stream hanging
            yield _sse({"type": "error", "message": f"Agent failed: {e}"})

        # Persist the assistant turn (figures rehydrated to Figure objects so
        # db.add_message serializes them the same way the Streamlit app did).
        if final_text:
            payload = {}
            if done:
                payload = {
                    "figures": [pio.from_json(json.dumps(f)) for f in done.get("figures", [])],
                    "wikipedia_urls": done.get("wikipedia", []),
                    "pubmed_urls": done.get("pubmed", []),
                    "ncbi_urls": done.get("ncbi", []),
                    "executed_codes": done.get("executed_codes", []),
                }
            db.add_message(conn, cid, "assistant", final_text, payload)
            db.touch_conversation(conn, cid)

        yield _sse({"type": "saved", "conversation_id": cid})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
