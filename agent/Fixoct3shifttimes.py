"""
One-time fix: this morning's Recurring Shift Generator run (2026-10-02 ~18:08 UTC)
wrote each shift's Bubble 'date' field using a fixed +10 offset, but the target
dates it generated (next Monday onward) fall after Melbourne's DST transition and
should have used +11 — so every one of those shifts displays 1 hour later than
intended (1am instead of midnight).

This re-sends the correct 'date' value (using the proper Melbourne offset for each
shift's own date) to every Bubble Shift record created in that run. Nothing else
about the shift is touched.

Run once:  python agent/fix_oct3_shift_times.py
"""

import os
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from dotenv import load_dotenv
import requests

import db

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fix-oct3-shift-times")

BUBBLE_BASE = "https://knightingale.com.au/api/1.1/obj"
BUBBLE_TOKEN = os.environ["BUBBLE_API_TOKEN"]
BUBBLE_HEADERS = {"Authorization": f"Bearer {BUBBLE_TOKEN}"}

# The faulty run's actual window, from its Railway log timestamps.
WINDOW_START = "2026-10-02T17:00:00+00:00"
WINDOW_END = "2026-10-02T19:00:00+00:00"


def melbourne_utc_offset_hours(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(
        hour=0, minute=0, tzinfo=ZoneInfo("Australia/Melbourne")
    )
    return int(dt.utcoffset().total_seconds() // 3600)


def get_affected_shifts() -> list[dict]:
    client = db.get_client()
    r = (
        client.table("shifts")
        .select("id, date, bubble_shift_id")
        .not_.is_("recurring_template_id", "null")
        .gte("created_at", WINDOW_START)
        .lte("created_at", WINDOW_END)
        .execute()
    )
    return r.data or []


def fix_shift(bubble_shift_id: str, date_str: str) -> bool:
    offset = melbourne_utc_offset_hours(date_str)
    try:
        r = requests.patch(
            f"{BUBBLE_BASE}/shift/{bubble_shift_id}",
            headers=BUBBLE_HEADERS,
            json={"date": f"{date_str}T00:00:00+{offset:02d}:00"},
            timeout=15,
        )
        r.raise_for_status()
        return True
    except Exception:
        logger.exception("Failed to fix shift %s", bubble_shift_id)
        return False


def main():
    shifts = get_affected_shifts()
    logger.info("Found %d affected shifts", len(shifts))
    fixed = 0
    for s in shifts:
        if not s.get("bubble_shift_id"):
            continue
        if fix_shift(s["bubble_shift_id"], s["date"]):
            fixed += 1
            logger.info("Fixed shift %s (%s)", s["bubble_shift_id"], s["date"])
    logger.info("Done: %d/%d fixed", fixed, len(shifts))


if __name__ == "__main__":
    main()
