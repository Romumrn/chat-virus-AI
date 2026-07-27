"""backend/routers/admin_routes.py — ADMIN-only platform management: list /
search users, change roles, delete users, global stats, and reading any user's
conversations. Mounted at /api/admin; every route requires the 'admin' role."""

from fastapi import APIRouter, Depends, HTTPException, status

import db
from backend import auth
from backend.schemas import UserOut, RoleUpdateIn

router = APIRouter(prefix="/api/admin", tags=["admin"])
_admin = auth.require_role("admin")


@router.get("/stats")
def stats(_: dict = Depends(_admin)):
    return db.get_stats(db.get_conn())


@router.get("/users", response_model=list[UserOut])
def list_users(q: str = "", _: dict = Depends(_admin)):
    conn = db.get_conn()
    return db.search_users(conn, q) if q else db.list_users_with_counts(conn)


@router.put("/users/{email}/role", response_model=UserOut)
def set_user_role(email: str, body: RoleUpdateIn, admin: dict = Depends(_admin)):
    conn = db.get_conn()
    email = email.lower()
    if not auth.is_valid_role(body.role):
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            f"Invalid role. Must be one of: user, dev, admin.")
    if db.get_user(conn, email) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    # Guard against an admin locking themselves out of admin.
    if email == admin["email"] and body.role != "admin":
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "You cannot remove your own admin role.")
    db.set_role(conn, email, body.role)
    return next(u for u in db.list_users_with_counts(conn) if u["email"] == email)


@router.delete("/users/{email}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(email: str, admin: dict = Depends(_admin)):
    conn = db.get_conn()
    email = email.lower()
    if db.get_user(conn, email) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    if email == admin["email"]:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "You cannot delete your own account.")
    db.delete_user(conn, email)


@router.get("/users/{email}/conversations")
def user_conversations(email: str, _: dict = Depends(_admin)):
    conn = db.get_conn()
    email = email.lower()
    if db.get_user(conn, email) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    return db.list_conversations(conn, email)


@router.get("/conversations/{conversation_id}/messages")
def read_any_conversation(conversation_id: int, _: dict = Depends(_admin)):
    conn = db.get_conn()
    conv = db.get_conversation(conn, conversation_id)
    if conv is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")
    return db.list_messages_json(conn, conversation_id)
