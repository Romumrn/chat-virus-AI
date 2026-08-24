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
    deaths_pct: float = Field(default=0.0, ge=0, le=1)
    dead: int = Field(ge=0)
    won: bool = False
    vaccine_deployed: bool = False


class ScoreRow(BaseModel):
    name: str
    virus: str
    days: int
    infected_pct: float
    dead: int
    score: int


class ScoreBreakdown(BaseModel):
    coverage_pts: int
    mortality_pts: int
    speed_pts: int
    vaccine_penalty: bool
    total: int


class ScoreOut(BaseModel):
    score: int
    breakdown: ScoreBreakdown
    leaderboard: list[ScoreRow]


# Score weights (kept here so the end-screen breakdown mirrors them exactly).
# Mortality is the goal: it dwarfs coverage and speed.
COVERAGE_MAX = 500  # points for infecting 100% of the world (secondary)
MORTALITY_MAX = 8000  # points for killing 100% of the world (dominant)
SPEED_MAX = 400  # points for an instant pandemic (1 pt per day saved, cap 400d)
VACCINE_MULT = 0.4  # score kept if the vaccine was deployed (a setback)


def _score_breakdown(body: "ScoreIn") -> ScoreBreakdown:
    coverage_pts = round(body.infected_pct * COVERAGE_MAX)
    mortality_pts = round(body.deaths_pct * MORTALITY_MAX)
    speed_pts = max(0, SPEED_MAX - body.days)
    subtotal = coverage_pts + mortality_pts + speed_pts
    total = round(subtotal * VACCINE_MULT) if body.vaccine_deployed else subtotal
    return ScoreBreakdown(
        coverage_pts=coverage_pts,
        mortality_pts=mortality_pts,
        speed_pts=speed_pts,
        vaccine_penalty=body.vaccine_deployed,
        total=total,
    )


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
    breakdown = _score_breakdown(body)
    db.add_helper_score(
        conn,
        user_email=user["email"],
        virus=body.virus,
        days=body.days,
        infected_pct=body.infected_pct,
        dead=body.dead,
        won=body.won,
        score=breakdown.total,
    )
    return ScoreOut(score=breakdown.total, breakdown=breakdown, leaderboard=_leaderboard(conn))


@router.get("/leaderboard", response_model=list[ScoreRow])
def leaderboard(user: dict = Depends(auth.get_current_user)):
    conn = db.get_conn()
    return _leaderboard(conn)
