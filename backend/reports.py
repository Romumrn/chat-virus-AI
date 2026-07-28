"""backend/reports.py — user-submitted "Report an error" feedback.

Each report is stored in the database (db.error_reports: id, user, date, status
open/in_progress/done) so devs can triage it from the Dev page. A JSON copy is
also written under logs/error_reports/ as a backup, capturing the question, the
model's answer, any executed code, the user's comment, and a tail of the current
month's agent log for context.
"""

import json
import os
from datetime import datetime

import db
from config import LOG_DIR
from logging_utils import setup_logger

logger = setup_logger(LOG_DIR)

REPORT_DIR = os.path.join(LOG_DIR, "error_reports")


def _recent_logs(timestamp: datetime, n: int = 200) -> list[str]:
    """Tail of the current month's agent log, for debugging context."""
    log_filename = os.path.join(LOG_DIR, f"agent_{timestamp.strftime('%Y-%m')}.log")
    if not os.path.exists(log_filename):
        return []
    try:
        with open(log_filename, "r", encoding="utf-8") as f:
            return [line.rstrip() for line in f.readlines()[-n:]]
    except Exception:
        return ["[Could not read log file]"]


def save_error_report(
    question: str,
    answer: str,
    executed_codes: list,
    comment: str = "",
    user_email: str | None = None,
) -> int:
    """Persist an error report (DB + JSON backup) and return its DB id."""
    timestamp = datetime.now()
    related_logs = _recent_logs(timestamp)

    # Primary store: the database, so it shows up in the Dev triage view.
    report_id = db.create_error_report(
        db.get_conn(),
        user_email=user_email,
        question=question,
        answer=answer,
        executed_codes=executed_codes,
        comment=comment,
        recent_logs=related_logs,
    )

    # Backup: a self-contained JSON file (same shape as the old Streamlit app).
    try:
        os.makedirs(REPORT_DIR, exist_ok=True)
        report_path = os.path.join(
            REPORT_DIR, f"report_{timestamp.strftime('%Y%m%d_%H%M%S')}_{report_id}.json"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "id": report_id,
                    "timestamp": timestamp.isoformat(),
                    "user_email": user_email,
                    "user_comment": comment,
                    "question": question,
                    "answer": answer,
                    "executed_codes": executed_codes,
                    "recent_logs": related_logs,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception as e:
        logger.warning(f"ERROR_REPORT | JSON backup failed: {e}")

    logger.warning(
        f"ERROR_REPORT | #{report_id} saved | user={user_email} | comment={comment!r}"
    )
    return report_id
