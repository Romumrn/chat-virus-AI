"""backend/routers/auth_routes.py — registration, login, current-user, and
self-service password change. Mounted at /api/auth."""

import os

from fastapi import APIRouter, Depends, HTTPException, status

import db
from config import _admin_emails
from backend import auth
from backend.schemas import (
    RegisterIn, LoginIn, TokenOut, PasswordChangeIn, UserOut,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _token_response(user: dict) -> TokenOut:
    token = auth.create_access_token(user["email"], user["role"])
    return TokenOut(
        access_token=token,
        role=user["role"],
        email=user["email"],
        first_name=user.get("first_name") or "",
        last_name=user.get("last_name") or "",
    )


@router.post("/register", response_model=TokenOut, status_code=status.HTTP_201_CREATED)
def register(body: RegisterIn):
    """Create an account and return a JWT so the user is logged in immediately.
    New accounts get the 'admin' role only if their email is in ADMIN_EMAILS,
    else 'user'."""
    conn = db.get_conn()

    expected_code = os.environ.get("REGISTRATION_CODE", "").strip()
    email = (body.email or "").strip().lower()
    first = (body.first_name or "").strip()
    last = (body.last_name or "").strip()

    if not (first and last and email and body.password):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "All fields are required.")
    if expected_code and (body.registration_code or "").strip() != expected_code:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "Invalid registration code — ask the Virome@t team for the current one.")

    err = auth.email_problem(email)
    if err:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, err)
    err = auth.password_problem(body.password)
    if err:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, err)

    if db.get_user(conn, email) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists.")

    role = "admin" if email in _admin_emails() else "user"
    db.create_user(conn, email, first, last, auth.hash_password(body.password), role)
    user = db.get_user(conn, email)
    return _token_response(user)


@router.post("/login", response_model=TokenOut)
def login(body: LoginIn):
    conn = db.get_conn()
    email = (body.email or "").strip().lower()
    user = db.get_user(conn, email)
    if user is None or not auth.verify_password(body.password, user.get("password_hash", "")):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    db.set_last_login(conn, email)
    return _token_response(user)


@router.get("/me", response_model=UserOut)
def me(user: dict = Depends(auth.get_current_user)):
    return UserOut(**user)


@router.put("/me/password", status_code=status.HTTP_204_NO_CONTENT)
def change_password(body: PasswordChangeIn, user: dict = Depends(auth.get_current_user)):
    conn = db.get_conn()
    full = db.get_user(conn, user["email"])
    if not auth.verify_password(body.current_password, full.get("password_hash", "")):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Current password is incorrect.")
    err = auth.password_problem(body.new_password)
    if err:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, err)
    db.update_password(conn, user["email"], auth.hash_password(body.new_password))
