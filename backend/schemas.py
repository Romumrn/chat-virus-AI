"""backend/schemas.py — Pydantic request/response models for the API."""

from pydantic import BaseModel, Field


# ==================== AUTH ====================

class RegisterIn(BaseModel):
    first_name: str
    last_name: str
    email: str
    password: str
    registration_code: str | None = None


class LoginIn(BaseModel):
    email: str
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    email: str
    first_name: str
    last_name: str


class PasswordChangeIn(BaseModel):
    current_password: str
    new_password: str


class UserOut(BaseModel):
    email: str
    first_name: str | None = None
    last_name: str | None = None
    role: str
    created_at: str | None = None
    last_login: str | None = None
    n_conversations: int | None = None


# ==================== CHAT / CONVERSATIONS ====================

class ChatIn(BaseModel):
    message: str
    conversation_id: int | None = None
    # Expert-mode overrides (DEV/ADMIN only; ignored/clamped for plain users).
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tool_calls: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    max_completion_tokens: int | None = None
    parallel_tool_calls: bool | None = None
    max_context_turns: int | None = None
    preview_rows: int | None = None
    wikipedia_limit: int | None = None
    max_tool_content: int | None = None


class ConversationOut(BaseModel):
    id: int
    title: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class ConversationCreateIn(BaseModel):
    title: str = "New conversation"


class ConversationRenameIn(BaseModel):
    title: str


class ErrorReportIn(BaseModel):
    """A user's "Report an error" submission for one assistant answer."""
    question: str = ""
    answer: str = ""
    executed_codes: list[str] = []
    comment: str = ""


# ==================== ADMIN / DEV ====================

class RoleUpdateIn(BaseModel):
    role: str = Field(..., description="one of: user, dev, admin")


class McpToolCallIn(BaseModel):
    name: str
    arguments: dict = {}
