"""
backend/auth.py — authentication for the FastAPI backend.

Stateless JWT sessions with bcrypt as the password engine.

  - Passwords: bcrypt hash / verify.
  - Sessions: a signed JWT (sub=email, role) issued at login, sent by the
    React app as `Authorization: Bearer <token>`.
  - Guards: get_current_user decodes the token and loads the user; require_role
    builds a dependency enforcing a minimum privilege level (user<dev<admin).
  - Registration policy: password rules + institutional-email blocklist.
"""

import re
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import db
from config import (
    JWT_SECRET, JWT_ALGORITHM, JWT_EXPIRE_MIN,
    VALID_ROLES, ROLE_LEVEL,
)

_bearer = HTTPBearer(auto_error=False)


# ==================== PASSWORDS ====================

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    if not password_hash:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# ==================== JWT ====================

def create_access_token(email: str, role: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": email,
        "role": role,
        "iat": now,
        "exp": now + timedelta(minutes=JWT_EXPIRE_MIN),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )


# ==================== FASTAPI DEPENDENCIES ====================

def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict:
    """Resolve the Bearer token to the current user record (without the hash).
    Raises 401 if missing/invalid, or if the user no longer exists."""
    if creds is None or not creds.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated"
        )
    payload = _decode_token(creds.credentials)
    email = payload.get("sub")
    conn = db.get_conn()
    user = db.get_user(conn, email) if email else None
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User no longer exists"
        )
    user.pop("password_hash", None)
    return user


def require_role(min_role: str):
    """Dependency factory: require the current user's role to be at least
    `min_role` (user < dev < admin). Returns the user on success, else 403."""
    threshold = ROLE_LEVEL[min_role]

    def _guard(user: dict = Depends(get_current_user)) -> dict:
        if ROLE_LEVEL.get(user.get("role", "user"), 0) < threshold:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires '{min_role}' role or higher",
            )
        return user

    return _guard


# ==================== REGISTRATION POLICY ====================

_BLOCKED_EMAIL_DOMAINS = {
    "gmail.com", "googlemail.com",
    "outlook.com", "outlook.fr", "hotmail.com", "hotmail.fr", "live.com", "live.fr", "msn.com",
    "yahoo.com", "yahoo.fr", "ymail.com",
    "icloud.com", "me.com", "mac.com",
    "aol.com",
    "proton.me", "protonmail.com",
    "gmx.com", "gmx.fr", "gmx.de",
    "orange.fr", "wanadoo.fr", "free.fr", "sfr.fr", "laposte.net", "bbox.fr",
    "yandex.com", "yandex.ru", "mail.com", "zoho.com",
}

_PASSWORD_RULES = [
    (lambda p: len(p) >= 12, "Password must be at least 12 characters long."),
    (lambda p: re.search(r"[a-z]", p) is not None, "Password must contain at least 1 lowercase letter."),
    (lambda p: re.search(r"[A-Z]", p) is not None, "Password must contain at least 1 uppercase letter."),
    (lambda p: re.search(r"[0-9]", p) is not None, "Password must contain at least 1 digit."),
    (lambda p: re.search(r"[^A-Za-z0-9]", p) is not None,
     "Password must contain at least 1 special character (e.g. ! ? @ # …)."),
]


def password_problem(password: str) -> str | None:
    """Human-readable reason the password is unacceptable, or None if OK."""
    for check, message in _PASSWORD_RULES:
        if not check(password):
            return message
    return None


def email_problem(email: str) -> str | None:
    """Validate the email shape and reject free-webmail domains (institutional
    addresses only). Returns None if acceptable."""
    if "@" not in email or "." not in email.split("@")[-1]:
        return "Please enter a valid email address."
    domain = email.split("@")[-1]
    if domain in _BLOCKED_EMAIL_DOMAINS:
        return (
            "Please register with an institutional email address "
            "(university or research lab) — free webmail providers are not accepted."
        )
    return None


def is_valid_role(role: str) -> bool:
    return role in VALID_ROLES
