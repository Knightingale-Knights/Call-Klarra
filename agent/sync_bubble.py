"""
Bubble -> Supabase sync.

Pulls active carers and their availability from Bubble's Data API and upserts them
into Supabase so Klarra works from real, current data.

  - nurses:    User where account type=carer, active=true  -> nurses (+ approvals)
  - availability: Availability where available=true        -> availability
  - shifts:    Shift (facility shifts)                      -> shifts

INCREMENTAL: after the first run, each run only fetches records Bubble says were
modified since the last SUCCESSFUL run (Bubble's built-in "Modified Date"), instead
of re-reading and rewriting everything. The time of the last successful run is kept
in the Supabase table sync_state. Rules that keep this safe:

  - The run start time is saved only after all three steps finish, so a crash means
    the next run retries from the old point (nothing is skipped).
  - 10 minutes of overlap is added, so a record edited while a run was in progress
    is not missed.
  - A FULL sync (everything, as before) runs on the very first run, whenever the
    last full sync is over 7 days old, and on demand (see below). The weekly full
    run catches anything an incremental run can't see, e.g. a carer who becomes
    eligible later while their older availability records are unchanged.
  - If the sync_state table is missing, every run is simply a full sync.

Force a full sync:  FULL_SYNC=1 environment variable, or:  python agent/sync_bubble.py --full

Run on demand:   python agent/sync_bubble.py
"""

import os
import sys
import logging
import requests
from datetime import datetime, timedelta, timezone

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

FULL_SYNC_EVERY_DAYS = 7
OVERLAP_MINUTES = 10
STATE_LAST_RUN = "bubble_sync_last_run"
STATE_LAST_FULL = "bubble_sync_last_full"

# Roles we staff, most senior first. A carer is stored under ONE role - the highest
# they hold - so someone qualified as both EN and PCA is only offered EN shifts.
# If dual-qualified carers need to see both, this becomes a roles[] column and a
# change to get_candidate_pool; single role is deliberate for now.
ROLE_PRIORITY = ["RN", "EN", "PCA", "DSW"]

# How Bubble's free-text role values map onto the four above. Anything not listed
# here is not a staffable role for Klarra (Chef, Admin, etc.) and the carer is skipped.
ROLE_ALIASES = {
    "RN": "RN",
    "REGISTERED NURSE": "RN",
    "EN": "EN",
    "ENROLLED NURSE": "EN",
    "PCA": "PCA",
    "PERSONAL CARE ASSISTANT": "PCA",
    "PERSONAL CARE WORKER": "PCA",
    "PCW": "PCA",
    "AIN": "PCA",
    "DSW": "DSW",
    "DISABILITY SUPPORT WORKER": "DSW",
    "SUPPORT WORKER": "DSW",
}

_unmapped_roles: set[str] = set()


# --- Sync state (when did we last sync successfully) ---

def get_state(name: str) -> datetime | None:
    try:
        r = (db.get_client().table("sync_state").select("value")
             .eq("name", name).limit(1).execute())
        if r.data:
            return datetime.fromisoformat(r.data[0]["value"])
    except Exception:
        logger.warning("Could not read sync_state (is the table created?) - "
                       "falling back to a full sync", exc_info=True)
    return None


def set_state(name: str, value: datetime) -> None:
    try:
        db.get_client().table("sync_state").upsert({
            "name": name,
            "value": value.isoformat(),
            "updated_at": "now()",
        }).execute()
    except Exception:
        logger.warning("Could not save sync_state - next run will be a full sync",
                       exc_info=True)


def decide_since(now: datetime) -> tuple[datetime | None, bool]:
    """Returns (since, is_full). since=None means fetch everything."""
    forced = "--full" in sys.argv or os.environ.get("FULL_SYNC", "").strip() in ("1", "true", "yes")
    last_run = get_state(STATE_LAST_RUN)
    last_full = get_state(STATE_LAST_FULL)
    if forced:
        logger.info("Full sync (forced)")
        return None, True
    if not last_run or not last_full:
        logger.info("Full sync (no previous successful run recorded)")
        return None, True
    if now - last_full > timedelta(days=FULL_SYNC_EVERY_DAYS):
        logger.info("Full sync (last full sync was over %d days ago)", FULL_SYNC_EVERY_DAYS)
        return None, True
    since = last_run - timedelta(minutes=OVERLAP_MINUTES)
    logger.info("Incremental sync: records modified since %s", since.isoformat())
    return since, False


def modified_since(since: datetime | None) -> list:
    """Bubble constraint list fragment for 'modified after since' (empty if full)."""
    if since is None:
        return []
    stamp = since.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    return [{"key": "Modified Date", "constraint_type": "greater than", "value": stamp}]


def pick_role(roles_list) -> str | None:
    """Take the highest-priority staffable role from Bubble's roles array.

    Returns None if the carer holds no role we staff - that carer is skipped, which
    is why PCAs and DSWs were previously absent from Supabase entirely: the old
    version only recognised EN and RN, so everyone else silently fell out of the sync.
    """
    if not roles_list:
        return None
    found = set()
    for raw in roles_list:
        key = str(raw).strip().upper()
        mapped = ROLE_ALIASES.get(key)
        if mapped:
            found.add(mapped)
        else:
            _unmapped_roles.add(str(raw).strip())
    for role in ROLE_PRIORITY:
        if role in found:
            return role
    return None


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


def sync_nurses(since: datetime | None = None):
    users = fetch_all("user", constraints=[
        {"key": "account type", "constraint_type": "equals", "value": "carer"},
        {"key": "active", "constraint_type": "equals", "value": "true"},
    ] + modified_since(since))
    logger.info("Fetched %d carers from Bubble", len(users))
    synced = 0
    skipped_no_role = 0
    skipped_no_phone = 0
    by_role: dict[str, int] = {}
    for u in users:
        role = pick_role(u.get("roles"))
        phone = u.get("phone number")
        if not role:
            skipped_no_role += 1
            continue
        if not phone:
            skipped_no_phone += 1
            continue
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
        by_role[role] = by_role.get(role, 0) + 1
        synced += 1
    logger.info("Synced %d nurses (%s)", synced,
                ", ".join(f"{r}: {c}" for r, c in sorted(by_role.items())) or "none")
    logger.info("Skipped: %d no staffable role, %d no phone",
                skipped_no_role, skipped_no_phone)
    if _unmapped_roles:
        logger.warning("Unmapped Bubble roles seen (add to ROLE_ALIASES if staffable): %s",
                       ", ".join(sorted(_unmapped_roles)))


def sync_availability(since: datetime | None = None):
    from datetime import date as _date
    today = _date.today().isoformat()
    avails = fetch_all("availability", constraints=[
        {"key": "available", "constraint_type": "equals", "value": "true"},
        {"key": "date", "constraint_type": "greater than", "value": today},
    ] + modified_since(since))
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
        db.upsert_availability(nid, date_only, shift, bubble_id=a["_id"])
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


def sync_shifts(since: datetime | None = None):
    shifts = fetch_all("shift", constraints=modified_since(since) or None)
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
    started = datetime.now(timezone.utc)
    since, is_full = decide_since(started)
    logger.info("Starting Bubble sync...")
    sync_nurses(since)
    sync_availability(since)
    sync_shifts(since)
    # Only reached if all three steps finished without an error.
    set_state(STATE_LAST_RUN, started)
    if is_full:
        set_state(STATE_LAST_FULL, started)
    logger.info("Done.")
