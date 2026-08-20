"""backend/routers/helper_routes.py — score persistence for the in-app helper
(the hidden contagion mini-game). Mounted at /api/helper.

Kept deliberately self-contained and neutrally named so it reads like ordinary
assistant telemetry. Two routes: submit a finished game (score is computed
server-side, never trusted from the client) and read the global top scores."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

import db
from backend import auth

router = APIRouter(prefix="/api/helper", tags=["helper"])


class ScoreIn(BaseModel):
    virus: str = Field(default="", max_length=40)
    days: int = Field(ge=0, le=100000)
    infected_pct: float = Field(ge=0, le=1)
    dead: int = Field(ge=0)
    won: bool = False


class ScoreRow(BaseModel):
    name: str
    virus: str
    days: int
    infected_pct: float
    dead: int
    score: int


class ScoreOut(BaseModel):
    score: int
    leaderboard: list[ScoreRow]


def _compute_score(days: int, infected_pct: float) -> int:
    """Reward covering the world both widely and fast: coverage scaled up by a
    speed bonus that decays with the number of days taken."""
    speed_bonus = max(0, 600 - days) * 2
    return round(infected_pct * (1000 + speed_bonus))


def _player_name(row) -> str:
    first = (row["first_name"] or "").strip()
    last = (row["last_name"] or "").strip()
    full = (first + " " + last).strip()
    if full:
        return full
    return (row["user_email"] or "Anonyme").split("@")[0]


def _leaderboard(conn, limit: int = 5) -> list[ScoreRow]:
    return [
        ScoreRow(
            name=_player_name(r),
            virus=r["virus"] or "",
            days=r["days"] or 0,
            infected_pct=r["infected_pct"] or 0.0,
            dead=r["dead"] or 0,
            score=r["score"] or 0,
        )
        for r in db.top_helper_scores(conn, limit)
    ]


@router.post("/score", response_model=ScoreOut)
def submit_score(body: ScoreIn, user: dict = Depends(auth.get_current_user)):
    conn = db.get_conn()
    score = _compute_score(body.days, body.infected_pct)
    db.add_helper_score(
        conn,
        user_email=user["email"],
        virus=body.virus,
        days=body.days,
        infected_pct=body.infected_pct,
        dead=body.dead,
        won=body.won,
        score=score,
    )
    return ScoreOut(score=score, leaderboard=_leaderboard(conn))


@router.get("/leaderboard", response_model=list[ScoreRow])
def leaderboard(user: dict = Depends(auth.get_current_user)):
    conn = db.get_conn()
    return _leaderboard(conn)
