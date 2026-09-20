"""
Shift sync webhook — receives a one-off shift created/edited in Bubble and pushes it
into Supabase, keyed on the shift's own Bubble _id so an edit updates instead of
duplicating.

If the shift is flagged recurring, also finds or creates a matching
recurring_shift_templates row (participant + nurse + weekday + start time). On
creation, the template also captures the fixed billing/address details (coordinator,
participant address, NDIS code, hours, rate, wage, revenue) from THIS shift, since
Paul confirmed these stay the same for a given recurring shift going forward.

Either way, flips the nurse's Supabase availability row to 'assigned' for that
date/shift_type, then pushes that change back to Bubble's own Availability record
(available=false) so the Bubble UI stays in sync.

Bubble workflow contract (on shift created/edited), POST form fields:
  shift_bubble_id       - the Shift thing's own _id
  nurse_bubble_id       - the assigned carer's _id
  date                  - YYYY-MM-DD
  start_time            - Bubble's own numeric time format, e.g. 900, 1430
  end_time              - same numeric format
  status                - confirmed|completed|cancelled (Bubble's shift status)
  recurring             - "yes"/"no" (Bubble's own dropdown value) or "true"/"false"
  facility_slug         - one of the known facility slugs, OR
  participant_bubble_id - the Participant thing's _id (send exactly one of these two)

  Only needed when recurring is true (captured once, on the template):
  coordinator_bubble_id - the participant's coordinator's _id
  participant_address   - the participant's address, as text
  ndis_code_bubble_id   - the NDIS Pricing item's _id
  ndis_code_text        - the NDIS item's display code/name
  hours                 - hours for this shift (number)
  rate                  - the NDIS item's price (number)
  wage                  - the carer's pay rate (number)
  revenue               - the item's hourly revenue rate (number)

shift_type (Morning/Afternoon/Night) is derived from start_time here, so Bubble
doesn't need to compute or send it.

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


def num_to_hhmm(n) -> str:
    """Bubble's numeric time (900, 1430) -> 'HH:MM'."""
    n = int(n)
    h, m = n // 100, n % 100
    return f"{h:02d}:{m:02d}"


def shift_type_from_start(n) -> str:
    """Classify a shift by its start hour — same convention as sync_bubble.py."""
    h = int(n) // 100
    if h < 12:
        return "Morning"
    if h < 18:
        return "Afternoon"
    return "Night"


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


def _num(v):
    """Best-effort float parse for optional numeric form fields; None if blank/absent."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


@app.route("/shift-sync", methods=["POST"])
def shift_sync():
    f = request.form
    shift_bubble_id = f.get("shift_bubble_id")
    nurse_bubble_id = f.get("nurse_bubble_id")
    date = f.get("date")
    start_time_num = f.get("start_time")
    end_time_num = f.get("end_time")
    status = f.get("status", "confirmed")
    recurring = f.get("recurring", "").strip().lower() in ("true", "yes")
    facility_slug = f.get("facility_slug") or None
    participant_bubble_id = f.get("participant_bubble_id") or None

    if not (shift_bubble_id and nurse_bubble_id and date
            and start_time_num and end_time_num):
        return jsonify({"error": "missing required field"}), 400

    try:
        start_hhmm = num_to_hhmm(start_time_num)
        end_hhmm = num_to_hhmm(end_time_num)
    except (TypeError, ValueError):
        return jsonify({"error": "start_time/end_time must be numeric (e.g. 900)"}), 400

    shift_type = shift_type_from_start(start_time_num)

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
            start_time=start_hhmm,
            end_time=end_hhmm,
            coordinator_bubble_id=f.get("coordinator_bubble_id") or None,
            participant_address=f.get("participant_address") or None,
            ndis_code_bubble_id=f.get("ndis_code_bubble_id") or None,
            ndis_code_text=f.get("ndis_code_text") or None,
            hours=_num(f.get("hours")),
            rate=_num(f.get("rate")),
            wage=_num(f.get("wage")),
            revenue_rate=_num(f.get("revenue")),
        )

    start_ts, end_ts = build_timestamps(date, start_hhmm, end_hhmm)

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
