"""backend/routers/conversation_routes.py — a user's own conversations:
list / create / rename / delete, plus reading a conversation's messages.
Mounted at /api/conversations. Every route enforces ownership."""

from fastapi import APIRouter, Depends, HTTPException, status

import db
from backend import auth
from backend.schemas import (
    ConversationOut, ConversationCreateIn, ConversationRenameIn,
)

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


def _owned_or_404(conn, conversation_id: int, email: str) -> dict:
    conv = db.get_conversation(conn, conversation_id)
    if conv is None or conv["user_email"] != email.lower():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")
    return conv


@router.get("", response_model=list[ConversationOut])
def list_my_conversations(user: dict = Depends(auth.get_current_user)):
    conn = db.get_conn()
    return db.list_conversations(conn, user["email"])


@router.post("", response_model=ConversationOut, status_code=status.HTTP_201_CREATED)
def create_conversation(body: ConversationCreateIn, user: dict = Depends(auth.get_current_user)):
    conn = db.get_conn()
    cid = db.create_conversation(conn, user["email"], body.title or "New conversation")
    return db.get_conversation(conn, cid)


@router.get("/{conversation_id}/messages")
def get_messages(conversation_id: int, user: dict = Depends(auth.get_current_user)):
    conn = db.get_conn()
    _owned_or_404(conn, conversation_id, user["email"])
    return db.list_messages_json(conn, conversation_id)


@router.patch("/{conversation_id}", response_model=ConversationOut)
def rename_conversation(
    conversation_id: int, body: ConversationRenameIn,
    user: dict = Depends(auth.get_current_user),
):
    conn = db.get_conn()
    _owned_or_404(conn, conversation_id, user["email"])
    db.rename_conversation(conn, conversation_id, body.title)
    return db.get_conversation(conn, conversation_id)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: int, user: dict = Depends(auth.get_current_user)):
    conn = db.get_conn()
    _owned_or_404(conn, conversation_id, user["email"])
    db.delete_conversation(conn, conversation_id)
