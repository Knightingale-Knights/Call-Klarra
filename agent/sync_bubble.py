"""
Bubble -> Supabase sync.

Pulls active carers and their availability from Bubble's Data API and upserts them
into Supabase so Klarra works from real, current data.

  - nurses:    User where account type=carer, active=true  -> nurses (+ approvals)
  - availability: Availability where available=true        -> availability

Run on demand:   python agent/sync_bubble.py
(Later: schedule it, or trigger per-signup from a Bubble workflow.)
"""

import os
import logging
import requests

import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from dotenv import load_dotenv
import db

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bubble-sync")

BASE = "https://knightingale.com.au/api/1.1/obj"
TOKEN = os.environ["BUBBLE_API_TOKEN"]
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


def fetch_all(datatype: str, constraints: list | None = None) -> list[dict]:
    """Page through a Bubble data type, respecting its 100-per-call limit."""
    results, cursor = [], 0
    params = {"limit": 100}
    if constraints:
        import json
        params["constraints"] = json.dumps(constraints)
    while True:
        params["cursor"] = cursor
        r = requests.get(f"{BASE}/{datatype}", headers=HEADERS, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()["response"]
        batch = payload.get("results", [])
        results.extend(batch)
        if payload.get("remaining", 0) <= 0:
            break
        cursor += len(batch)
    return results


def pick_role(roles_list) -> str | None:
    """Take EN or RN from the roles array; ignore others (DSW, AIN, etc.)."""
    if not roles_list:
        return None
    upper = [str(r).upper() for r in roles_list]
    if "RN" in upper:
        return "RN"
    if "EN" in upper:
        return "EN"
    return None


def sync_nurses():
    users = fetch_all("user", constraints=[
        {"key": "account type", "constraint_type": "equals", "value": "carer"},
        {"key": "active", "constraint_type": "equals", "value": "true"},
    ])
    logger.info("Fetched %d carers from Bubble", len(users))
    synced = 0
    for u in users:
        role = pick_role(u.get("roles"))
        phone = u.get("phone number")
        if not role or not phone:
            continue  # need a callable phone and a staffable role
        addr = (u.get("address") or {}).get("address") if isinstance(u.get("address"), dict) else None
        nid = db.upsert_nurse(
            bubble_id=u["_id"],
            first_name=u.get("first name", ""),
            last_name=u.get("last name", ""),
            phone=phone,
            role=role,
            address=addr,
        )
        slugs = [db.FACILITY_NAME_TO_SLUG[n] for n in (u.get("work locations") or [])
                 if n in db.FACILITY_NAME_TO_SLUG]
        db.set_nurse_approvals(nid, slugs)
        synced += 1
    logger.info("Synced %d nurses", synced)


def sync_availability():
    from datetime import date as _date
    today = _date.today().isoformat()
    avails = fetch_all("availability", constraints=[
        {"key": "available", "constraint_type": "equals", "value": "true"},
        {"key": "date", "constraint_type": "greater than", "value": today},
    ])
    logger.info("Fetched %d future availability records from Bubble", len(avails))
    synced = 0
    for a in avails:
        carer_bubble_id = a.get("carer")
        date = a.get("date")
        shift = a.get("shift")  # 'Morning' | 'Afternoon' | 'Night'
        if not (carer_bubble_id and date and shift):
            continue
        nid = db.nurse_id_by_bubble(carer_bubble_id)
        if not nid:
            continue  # nurse not synced (inactive / no role)
        date_only = date[:10]  # ISO -> YYYY-MM-DD
        db.upsert_availability(nid, date_only, shift)
        synced += 1
    logger.info("Synced %d availability records", synced)


def num_to_time(n) -> str:
    """Bubble time number (e.g. 900, 1430) -> 'HH:MM:SS'."""
    n = int(n)
    h, m = n // 100, n % 100
    return f"{h:02d}:{m:02d}:00"


def shift_type_from_start(n) -> str:
    """Classify a shift by its start hour."""
    h = int(n) // 100
    if h < 12:
        return "Morning"
    if h < 18:
        return "Afternoon"
    return "Night"


def sync_shifts():
    shifts = fetch_all("shift")
    logger.info("Fetched %d shifts from Bubble", len(shifts))
    synced = 0
    for s in shifts:
        carer_id = s.get("carer")
        loc_id = s.get("location")
        date = s.get("date")
        st = s.get("start time")
        et = s.get("end time")
        if not (carer_id and loc_id and date and st is not None and et is not None):
            continue
        slug = db.LOCATION_ID_TO_SLUG.get(loc_id)
        if not slug:
            continue  # location not one of our facilities
        nid = db.nurse_id_by_bubble(carer_id)
        fid = db.facility_id_by_slug(slug)
        if not (nid and fid):
            continue
        date_only = date[:10]
        start_ts = f"{date_only} {num_to_time(st)}+10"
        # Overnight shift: if end time is earlier than start, it ends the next day.
        if int(et) <= int(st):
            from datetime import date as _d, timedelta
            y, m, d = map(int, date_only.split("-"))
            end_date = (_d(y, m, d) + timedelta(days=1)).isoformat()
        else:
            end_date = date_only
        end_ts = f"{end_date} {num_to_time(et)}+10"
        status = "cancelled" if s.get("cancelled") else (
            "completed" if s.get("accepted") else "confirmed")
        db.upsert_shift(s["_id"], nid, fid, date_only,
                        shift_type_from_start(st), start_ts, end_ts, status)
        synced += 1
    logger.info("Synced %d shifts", synced)


if __name__ == "__main__":
    logger.info("Starting Bubble sync...")
    sync_nurses()
    sync_availability()
    sync_shifts()
    logger.info("Done.")
