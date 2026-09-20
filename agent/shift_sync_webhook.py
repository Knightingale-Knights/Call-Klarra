"""
Shift sync webhook — receives a one-off shift created/edited in Bubble and pushes it
into Supabase, keyed on the shift's own Bubble _id so an edit updates instead of
duplicating.

If the shift is flagged recurring, also finds or creates a matching
recurring_shift_templates row (participant + nurse + weekday + start time).

Either way, flips the nurse's Supabase availability row to 'assigned' for that
date/shift_type, then pushes that change back to Bubble's own Availability record
(available=false) so the Bubble UI stays in sync.

Bubble workflow contract (on shift created/edited), POST form fields:
  shift_bubble_id      - the Shift thing's own _id
  nurse_bubble_id       - the assigned carer's _id
  date                  - YYYY-MM-DD
  shift_type            - Morning|Afternoon|Night
  start_time            - HH:MM (24h)
  end_time              - HH:MM (24h)
  status                - confirmed|completed|cancelled (Bubble's shift status)
  recurring             - "true" or "false"
  facility_slug         - one of the known facility slugs, OR
  participant_bubble_id - the Participant thing's _id (send exactly one of these two)

Run:  python agent/shift_sync_webhook.py
"""

import os
import logging
from datetime import date as _date, datetime, timedelta

import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from dotenv import load_dotenv
from flask import Flask, request, jsonify
import requests

import db

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shift-sync")

app = Flask(__name__)

BUBBLE_BASE = "https://knightingale.com.au/api/1.1/obj"
BUBBLE_TOKEN = os.environ["BUBBLE_API_TOKEN"]
BUBBLE_HEADERS = {"Authorization": f"Bearer {BUBBLE_TOKEN}"}


def build_timestamps(date_str: str, start_hhmm: str, end_hhmm: str) -> tuple[str, str]:
    """Turn a date + two HH:MM times into full timestamptz strings (Melbourne, +10),
    same convention sync_bubble.py uses. An overnight shift (end <= start) rolls the
    end timestamp to the next calendar day."""
    start_ts = f"{date_str} {start_hhmm}:00+10"
    if end_hhmm <= start_hhmm:
        y, m, d = map(int, date_str.split("-"))
        end_date = (_date(y, m, d) + timedelta(days=1)).isoformat()
    else:
        end_date = date_str
    end_ts = f"{end_date} {end_hhmm}:00+10"
    return start_ts, end_ts


def push_availability_to_bubble(availability_bubble_id: str, available: bool) -> None:
    """PATCH the Availability record's own 'available' field back in Bubble."""
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


@app.route("/shift-sync", methods=["POST"])
def shift_sync():
    f = request.form
    shift_bubble_id = f.get("shift_bubble_id")
    nurse_bubble_id = f.get("nurse_bubble_id")
    date = f.get("date")
    shift_type = f.get("shift_type")
    start_time = f.get("start_time")
    end_time = f.get("end_time")
    status = f.get("status", "confirmed")
    recurring = f.get("recurring", "false").strip().lower() == "true"
    facility_slug = f.get("facility_slug") or None
    participant_bubble_id = f.get("participant_bubble_id") or None

    if not (shift_bubble_id and nurse_bubble_id and date and shift_type
            and start_time and end_time):
        return jsonify({"error": "missing required field"}), 400

    nurse_id = db.nurse_id_by_bubble(nurse_bubble_id)
    if not nurse_id:
        return jsonify({"error": f"unknown nurse_bubble_id {nurse_bubble_id}"}), 404

    facility_id = db.facility_id_by_slug(facility_slug) if facility_slug else None
    participant_id = (db.participant_id_by_bubble(participant_bubble_id)
                      if participant_bubble_id else None)
    if not facility_id and not participant_id:
        return jsonify({"error": "must supply facility_slug or participant_bubble_id"}), 400

    recurring_template_id = None
    if recurring and participant_id:
        try:
            day_of_week = datetime.strptime(date, "%Y-%m-%d").weekday()
        except ValueError:
            return jsonify({"error": f"bad date {date}"}), 400
        nurse = db.get_nurse(nurse_id)
        recurring_template_id = db.find_or_create_recurring_template(
            participant_id=participant_id,
            nurse_id=nurse_id,
            role=(nurse or {}).get("role", ""),
            day_of_week=day_of_week,
            start_time=start_time,
            end_time=end_time,
        )

    start_ts, end_ts = build_timestamps(date, start_time, end_time)

    db.upsert_shift_from_push(
        bubble_shift_id=shift_bubble_id,
        nurse_id=nurse_id,
        date=date,
        shift_type=shift_type,
        start_time=start_ts,
        end_time=end_ts,
        status=status,
        facility_id=facility_id,
        participant_id=participant_id,
        recurring_template_id=recurring_template_id,
    )

    db.assign_availability(nurse_id, date, shift_type)
    avail_bubble_id = db.get_availability_bubble_id(nurse_id, date, shift_type)
    if avail_bubble_id:
        push_availability_to_bubble(avail_bubble_id, available=False)

    logger.info("Synced shift %s (nurse=%s %s %s)", shift_bubble_id, nurse_id, date, shift_type)
    return jsonify({"ok": True, "recurring_template_id": recurring_template_id})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))
