"""backend/reports.py — user-submitted "Report an error" feedback.

Ported verbatim (behaviour unchanged) from the Streamlit app's
save_error_report: each report is a JSON file under logs/error_reports/,
capturing the question, the model's answer, any executed code, the user's
comment, and a tail of the current month's agent log for context.
"""

import json
import os
from datetime import datetime

from config import LOG_DIR
from logging_utils import setup_logger

logger = setup_logger(LOG_DIR)

REPORT_DIR = os.path.join(LOG_DIR, "error_reports")


def save_error_report(
    question: str, answer: str, executed_codes: list, comment: str = ""
) -> str:
    """Persist an error report with debugging context and return its path."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    timestamp = datetime.now()
    ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORT_DIR, f"report_{ts_str}.json")

    log_filename = os.path.join(LOG_DIR, f"agent_{timestamp.strftime('%Y-%m')}.log")
    related_logs = []
    if os.path.exists(log_filename):
        try:
            with open(log_filename, "r", encoding="utf-8") as f:
                related_logs = [line.rstrip() for line in f.readlines()[-200:]]
        except Exception:
            related_logs = ["[Could not read log file]"]

    report = {
        "timestamp": timestamp.isoformat(),
        "user_comment": comment,
        "question": question,
        "answer": answer,
        "executed_codes": executed_codes,
        "recent_logs": related_logs,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.warning(f"ERROR_REPORT | Saved to {report_path} | comment={comment!r}")
    return report_path
