"""
Recurring shift generator — runs weekly. For every active recurring_shift_template,
creates that week's Shift in Bubble (with the same fields Paul's manual "Create a new
Shift" workflow sets), then pushes it into Supabase, keyed on the new shift's Bubble
_id, and flips the nurse's availability the same way shift_sync_webhook.py does.

Cadence: each template's NEXT shift date is always the last shift generated for that
template, plus 7 days — not "the matching weekday in whatever week the cron happens to
run" — so the schedule is anchored to actual shift history and stays correct even if
a run is skipped or run late. A template always has at least one shift (the one that
originally triggered it), so the "no shift yet" fallback below should be rare.

Idempotent: if a template already has a shift for its target date (e.g. the cron ran
twice, or was re-triggered manually), it's skipped rather than duplicated.

Run:  python agent/recurring_shift_generator.py
"""

import os
import logging
from datetime import date as _date, datetime, timedelta

import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from dotenv import load_dotenv
import requests

import db

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recurring-shift-generator")

BUBBLE_BASE = "https://knightingale.com.au/api/1.1/obj"
BUBBLE_TOKEN = os.environ["BUBBLE_API_TOKEN"]
BUBBLE_HEADERS = {"Authorization": f"Bearer {BUBBLE_TOKEN}"}


def hhmm(t) -> str:
    """Supabase returns a `time` column as 'HH:MM:SS' (or already 'HH:MM') —
    normalise to 'HH:MM'."""
    return str(t)[:5]


def hhmm_to_bubble_num(t) -> int:
    """'09:00' -> 900, '14:30' -> 1430 — matches sync_bubble.py's reading convention,
    so a shift this script creates reads back the same way the nightly sync expects."""
    h, m = hhmm(t).split(":")
    return int(h) * 100 + int(m)


def shift_type_from_start(t) -> str:
    h = int(hhmm(t).split(":")[0])
    if h < 12:
        return "Morning"
    if h < 18:
        return "Afternoon"
    return "Night"


def build_timestamps(date_str: str, start_hhmm: str, end_hhmm: str) -> tuple[str, str]:
    """Same convention as shift_sync_webhook.py / sync_bubble.py: Melbourne (+10),
    end rolls to the next day if it's an overnight shift."""
    start_ts = f"{date_str} {start_hhmm}:00+10"
    if end_hhmm <= start_hhmm:
        y, m, d = map(int, date_str.split("-"))
        end_date = (_date(y, m, d) + timedelta(days=1)).isoformat()
    else:
        end_date = date_str
    end_ts = f"{end_date} {end_hhmm}:00+10"
    return start_ts, end_ts


def push_availability_to_bubble(availability_bubble_id: str, available: bool) -> None:
    try:
        r = requests.patch(
            f"{BUBBLE_BASE}/availability/{availability_bubble_id}",
            headers=BUBBLE_HEADERS,
            json={"available": available},
            timeout=15,
        )
        r.raise_for_status()
    except Exception:
        logger.exception("Failed to push availability back to Bubble (%s)",
                         availability_bubble_id)


def next_target_date(template: dict) -> str:
    """This template's next shift date: last generated date + 7 days, or — only if
    it somehow has no shifts yet — the next occurrence of its weekday from today."""
    last = db.latest_shift_date_for_template(template["id"])
    if last:
        y, m, d = map(int, str(last)[:10].split("-"))
        return (_date(y, m, d) + timedelta(days=7)).isoformat()
    today = _date.today()
    days_ahead = (template["day_of_week"] - today.weekday()) % 7 or 7
    return (today + timedelta(days=days_ahead)).isoformat()


def create_bubble_shift(template: dict, target_date: str) -> str | None:
    """Create the Shift object in Bubble with the same fields as Paul's manual
    'Create a new Shift' workflow, using this template's stored, fixed values.
    Returns the new shift's Bubble _id, or None on failure."""
    carer_bubble_id = db.nurse_bubble_id(template["nurse_id"])
    participant_bid = db.participant_bubble_id(template["participant_id"])
    if not (carer_bubble_id and participant_bid):
        logger.error("Template %s: missing carer or participant Bubble id, skipping",
                     template["id"])
        return None

    hours = template.get("hours") or 0
    rate = template.get("rate") or 0
    wage_rate = template.get("wage") or 0
    revenue_rate = template.get("revenue_rate") or 0

    payload = {
        "accepted": "yes",
        "cancelled": "no",
        "attended": "no",
        "invoiced": "no",
        "carer": carer_bubble_id,
        "participant": participant_bid,
        "coordinator": template.get("coordinator_bubble_id"),
        "address": template.get("participant_address"),
        # Melbourne midnight, expressed with the same fixed +10 offset used
        # elsewhere in this codebase (ignores daylight saving, same as
        # build_timestamps) — sending UTC midnight here instead would display as
        # 10am/11am in Bubble, since Bubble shows dates in local time.
        "date": f"{target_date}T00:00:00+10:00",
        "start time": hhmm_to_bubble_num(template["start_time"]),
        "end time": hhmm_to_bubble_num(template["end_time"]),
        "hours": hours,
        "ndis": template.get("ndis_code_bubble_id"),
        "rate": rate,
        "fee": hours * rate,
        "wage": hours * wage_rate,
        "revenue": hours * revenue_rate,
        "roles": ["DSW"],
        # Assumption: Bubble's recurring field is a yes/no text value, same
        # convention as accepted/cancelled/attended/invoiced above.
        "recurring": "yes",
    }
    try:
        r = requests.post(f"{BUBBLE_BASE}/shift", headers=BUBBLE_HEADERS,
                          json=payload, timeout=20)
        if not r.ok:
            logger.error("Bubble rejected shift create for template %s: %s %s\nPayload: %s",
                        template["id"], r.status_code, r.text, payload)
            return None
        return r.json().get("id")
    except Exception:
        logger.exception("Failed to create Bubble shift for template %s", template["id"])
        return None


def generate_for_template(template: dict) -> int | None:
    """Returns the nurse_id if a shift was actually generated this run, else None
    (already existed for this date, or creation failed) — main() uses this to know
    which nurses to send the weekly roster-check text to."""
    target_date = next_target_date(template)

    if db.shift_exists_for_template(template["id"], target_date):
        logger.info("Template %s: shift for %s already exists, skipping",
                    template["id"], target_date)
        return None

    new_bubble_id = create_bubble_shift(template, target_date)
    if not new_bubble_id:
        return None

    start_ts, end_ts = build_timestamps(
        target_date, hhmm(template["start_time"]), hhmm(template["end_time"])
    )
    shift_type = shift_type_from_start(template["start_time"])

    db.upsert_shift_from_push(
        bubble_shift_id=new_bubble_id,
        nurse_id=template["nurse_id"],
        date=target_date,
        shift_type=shift_type,
        start_time=start_ts,
        end_time=end_ts,
        status="confirmed",
        participant_id=template["participant_id"],
        recurring_template_id=template["id"],
    )

    db.assign_availability(template["nurse_id"], target_date, shift_type)
    avail_bubble_id = db.get_availability_bubble_id(
        template["nurse_id"], target_date, shift_type
    )
    if avail_bubble_id:
        push_availability_to_bubble(avail_bubble_id, available=False)

    logger.info("Generated shift %s for template %s (%s %s)",
                new_bubble_id, template["id"], target_date, shift_type)
    return template["nurse_id"]


def send_roster_reminder(nurse_id: int) -> None:
    """One text per nurse, regardless of how many recurring shifts they got this
    run — asks them to check the roster and flag anything that looks wrong."""
    nurse = db.get_nurse(nurse_id)
    if not nurse or not nurse.get("phone"):
        logger.warning("Skipping roster reminder for nurse %s: no phone on file", nurse_id)
        return
    name = nurse.get("first_name") or "there"
    body = (f"Hi {name}, it's Klarra from Knightingale. Next week's roster is up. "
            f"Please check your shifts and let Paul know if anything looks off. "
            f"He will instruct me on how to fix it. If everything looks good, you "
            f"don't need to do anything. Have fun shifts \U0001F642")
    try:
        db.send_sms(nurse["phone"], body)
        logger.info("Sent roster reminder to nurse %s", nurse_id)
    except Exception:
        logger.exception("Failed to send roster reminder to nurse %s", nurse_id)


def main():
    templates = db.get_active_recurring_templates()
    logger.info("Found %d active recurring templates", len(templates))
    notified_nurses = set()
    for t in templates:
        try:
            nurse_id = generate_for_template(t)
            if nurse_id:
                notified_nurses.add(nurse_id)
        except Exception:
            logger.exception("Failed generating shift for template %s", t.get("id"))

    logger.info("Sending roster reminders to %d nurse(s)", len(notified_nurses))
    for nurse_id in notified_nurses:
        send_roster_reminder(nurse_id)


if __name__ == "__main__":
    main()
