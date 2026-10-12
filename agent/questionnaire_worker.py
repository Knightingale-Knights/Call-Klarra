"""
Questionnaire worker (Railway service, always on).

Every minute:
  1. Heads-up (5pm-9pm Melbourne, every 15 min, idempotent):
     for each NDIS participant with a shift tomorrow (first one starting 7am-7pm) and
     empty questionnaire fields, create a Questionnaire Session for that shift and text
     the carer a heads-up naming what's missing and when Klarra will call.
  2. Calls: for each Scheduled session whose call time has passed, check the shift is
     still on and the carer has checked in (Bubble "check in time" set), then dispatch
     the outbound agent (kind="questionnaire"). Not checked in: retry every 5 min until
     90 min after shift start, then mark Failed.

Env:
  QUESTIONNAIRE_ONLY_PARTICIPANT  optional Bubble user id: restrict to one participant (testing)
  QUESTIONNAIRE_TEST_NOW=1        on start, schedule a test call in 2 minutes (and skip the 5pm heads-up run)
  QUESTIONNAIRE_TEST_SHIFT        Bubble shift id to test with (else ONLY's next shift)
  QUESTIONNAIRE_SKIP_CHECKIN=1    don't wait for the carer's check in (testing)
  OUTBOUND_AGENT_NAME             which outbound agent to dispatch (default knightingale-outbound)
  QUESTIONNAIRE_BUBBLE_BASE       Bubble Data API base (dev: https://knightingale.com.au/version-test/api/1.1/obj)
  plus the usual BUBBLE_API_TOKEN, SUPABASE_*, TWILIO_*, LIVEKIT_*, KLARRA_MODE.

Run:  python agent/questionnaire_worker.py
"""

import os
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta, timezone

import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from dotenv import load_dotenv
from livekit import api

load_dotenv()

import db
import questionnaire as q

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("questionnaire-worker")
for noisy in ("hpack", "httpx", "httpcore", "h2"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

AGENT_NAME = os.environ.get("OUTBOUND_AGENT_NAME", "knightingale-outbound")
TEST_NOW = os.environ.get("QUESTIONNAIRE_TEST_NOW") == "1"
SKIP_CHECKIN = os.environ.get("QUESTIONNAIRE_SKIP_CHECKIN") == "1"
ONLY = os.environ.get("QUESTIONNAIRE_ONLY_PARTICIPANT", "").strip() or None
TEST_SHIFT = os.environ.get("QUESTIONNAIRE_TEST_SHIFT", "").strip() or None

HEADS_UP_START, HEADS_UP_END = 17, 21      # Melbourne hours
EARLIEST_START, LATEST_START = 7, 19        # only shifts starting in this window
CALL_OFFSET = timedelta(minutes=30)
CHECKIN_GRACE = timedelta(minutes=90)


def _cancelled(shift: dict) -> bool:
    return str(shift.get("cancelled", "")).lower() in ("yes", "true")


# ---------- 1. heads-up ----------

def tomorrows_participant_shifts() -> dict[str, dict]:
    """Earliest eligible shift tomorrow per participant: {participant_bid: shift}."""
    tomorrow = (q.melb_now() + timedelta(days=1)).date()
    lo = datetime(tomorrow.year, tomorrow.month, tomorrow.day, tzinfo=q.MELB) - timedelta(minutes=1)
    hi = lo + timedelta(days=1)
    cons = [
        {"key": "date", "constraint_type": "greater than", "value": lo.isoformat()},
        {"key": "date", "constraint_type": "less than", "value": hi.isoformat()},
        {"key": "participant", "constraint_type": "is_not_empty"},
    ]
    if ONLY:
        cons.append({"key": "participant", "constraint_type": "equals", "value": ONLY})
    best: dict[str, dict] = {}
    for s in q.bubble_get("shift", cons):
        if _cancelled(s) or not s.get("carer"):
            continue
        st = q.shift_start(s)
        if not st or not (EARLIEST_START <= st.hour < LATEST_START):
            continue
        p = s["participant"]
        if p not in best or st < q.shift_start(best[p]):
            best[p] = s
    return best


def session_exists_for_shift(shift_bid: str) -> bool:
    r = (db.get_client().table("questionnaire_sessions").select("id")
         .eq("shift_bubble_id", shift_bid).limit(1).execute())
    return bool(r.data)


def schedule_one(participant_bid: str, shift: dict, call_at: datetime | None = None,
                 when: str = "tomorrow") -> None:
    if session_exists_for_shift(shift["_id"]):
        logger.info("Session already exists for shift %s", shift["_id"])
        return
    nxt = q.next_section(participant_bid)
    if not nxt:
        return
    section, askable = nxt
    participant = q.bubble_user(participant_bid)
    carer = q.bubble_user(shift["carer"])
    phone = db.normalise_phone(carer.get("phone number"))
    if not phone:
        logger.warning("Carer %s has no phone, skipping", shift["carer"])
        return
    call_at = call_at or (q.shift_start(shift) + CALL_OFFSET)
    p_name = q.first_name(participant)
    now = datetime.now(timezone.utc).isoformat()

    bubble_id = q.bubble_create("questionnairesession", {
        "participant": participant_bid, "carer": shift["carer"], "shift": shift["_id"],
        "section": section, "status": "Scheduled",
        "scheduled call": call_at.isoformat(), "heads up sent": now,
    })
    db.get_client().table("questionnaire_sessions").insert({
        "bubble_id": bubble_id, "participant_bubble_id": participant_bid,
        "carer_bubble_id": shift["carer"], "carer_phone": phone, "section": section,
        "status": "Scheduled", "shift_bubble_id": shift["_id"],
        "scheduled_call": call_at.isoformat(), "heads_up_sent": now,
    }).execute()

    body = (f"Hi {q.first_name(carer)}, it's Klarra from Knightingale. I noticed we don't have "
            f"any information for {p_name} on {q.topics_phrase(section, askable)}. I'll give you "
            f"a call {when} at {q.pretty_time(call_at)} when you're with {p_name}. This way I "
            f"can ask you to ask {p_name}. Speak then.")
    db.send_sms(phone, body)
    logger.info("Heads-up sent: %s / %s / %s at %s", p_name, section, phone, call_at)


def run_heads_up() -> None:
    shifts = tomorrows_participant_shifts()
    logger.info("Heads-up pass: %d participants with shifts tomorrow", len(shifts))
    for pbid, shift in shifts.items():
        try:
            schedule_one(pbid, shift)
        except Exception:
            logger.exception("Heads-up failed for participant %s", pbid)


# ---------- 2. calls ----------

def due_sessions() -> list[dict]:
    now = datetime.now(timezone.utc).isoformat()
    r = (db.get_client().table("questionnaire_sessions").select("*")
         .eq("status", "Scheduled").lte("scheduled_call", now).execute())
    return r.data or []


def call_number(phone: str) -> str | None:
    """Same contact rules as SMS: dev redirects to the dev phone, mid only reaches testers."""
    phone = db.normalise_phone(phone)
    if db.DEV and phone not in db.dev_testers():
        return os.environ.get("KLARRA_DEV_PHONE")
    if db.MID and phone not in db.dev_testers():
        logger.warning("[MID] blocked questionnaire call to %s", phone)
        return None
    return phone


async def start_call(lk: api.LiveKitAPI, session: dict) -> None:
    shift = q.bubble_get_by_id("shift", session["shift_bubble_id"])
    if not shift or _cancelled(shift):
        q.update_session(session, status="Failed", skipped_fields="shift cancelled")
        return
    start = q.shift_start(shift)
    if not shift.get("check in time") and not SKIP_CHECKIN:
        if q.melb_now() > start + CHECKIN_GRACE:
            q.update_session(session, status="Failed", skipped_fields="carer never checked in")
        else:
            nxt = datetime.now(timezone.utc) + timedelta(minutes=5)
            db.get_client().table("questionnaire_sessions").update(
                {"scheduled_call": nxt.isoformat()}).eq("id", session["id"]).execute()
        return

    # Carer may have changed since the heads-up: call whoever is on the shift now.
    carer_bid = shift.get("carer") or session["carer_bubble_id"]
    carer = q.bubble_user(carer_bid)
    participant = q.bubble_user(session["participant_bubble_id"])
    phone = call_number(carer.get("phone number") or session.get("carer_phone"))
    if not phone:
        q.update_session(session, status="Failed", skipped_fields="call blocked by KLARRA_MODE")
        return

    section = session["section"]
    record = q.get_or_create_record(section, session["participant_bubble_id"])
    skipped = q.recently_skipped(session["participant_bubble_id"], section)
    fields = [k for k in q.missing_fields(section, record) if k not in skipped]
    if not fields:
        q.update_session(session, status="Completed",
                         completed=datetime.now(timezone.utc).isoformat())
        return

    meta = {
        "kind": "questionnaire", "phone": phone, "session_id": session["id"],
        "section": section, "fields": fields, "record_id": record["_id"],
        "participant_bubble_id": session["participant_bubble_id"],
        "participant_name": q.first_name(participant), "carer_name": q.first_name(carer),
    }
    room = f"questionnaire-{session['id'][:8]}-{int(time.time())}"
    q.update_session(session, status="Calling", carer_bubble_id=carer_bid,
                     started=datetime.now(timezone.utc).isoformat())
    await lk.agent_dispatch.create_dispatch(api.CreateAgentDispatchRequest(
        agent_name=AGENT_NAME, room=room, metadata=json.dumps(meta)))
    logger.info("Dispatched questionnaire call %s (%s, %d fields)", room, section, len(fields))


def schedule_test_now() -> None:
    """Testing: schedule TEST_SHIFT (or ONLY's next shift) with a call in 2 minutes."""
    global ONLY
    if TEST_SHIFT:
        shift = q.bubble_get_by_id("shift", TEST_SHIFT)
        if not shift or not shift.get("participant") or not shift.get("carer"):
            logger.warning("Test shift %s not found, or has no participant/carer", TEST_SHIFT)
            return
        ONLY = shift["participant"]
        logger.info("Test shift %s, participant %s", TEST_SHIFT, ONLY)
        schedule_one(ONLY, shift, call_at=q.melb_now() + timedelta(minutes=2), when="today")
        return
    if not ONLY:
        logger.warning("QUESTIONNAIRE_TEST_NOW needs QUESTIONNAIRE_ONLY_PARTICIPANT")
        return
    today = q.melb_now().date()
    lo = datetime(today.year, today.month, today.day, tzinfo=q.MELB) - timedelta(minutes=1)
    shifts = [s for s in q.bubble_get("shift", [
        {"key": "participant", "constraint_type": "equals", "value": ONLY},
        {"key": "date", "constraint_type": "greater than", "value": lo.isoformat()},
    ]) if not _cancelled(s) and s.get("carer") and q.shift_start(s)]
    if not shifts:
        logger.warning("Test: no upcoming shift for participant %s", ONLY)
        return
    shift = min(shifts, key=q.shift_start)
    call_at = q.melb_now() + timedelta(minutes=2)
    schedule_one(ONLY, shift, call_at=call_at, when="today")


async def main():
    lk = api.LiveKitAPI()
    last_heads_up = 0.0
    logger.info("Questionnaire worker started (KLARRA_MODE=%s, only=%s, agent=%s)",
                db.KLARRA_MODE, ONLY, AGENT_NAME)
    if TEST_NOW:
        try:
            schedule_test_now()
        except Exception:
            logger.exception("Test scheduling failed")
    while True:
        try:
            h = q.melb_now().hour
            if (not TEST_NOW and HEADS_UP_START <= h < HEADS_UP_END
                    and time.time() - last_heads_up > 900):
                run_heads_up()
                last_heads_up = time.time()
            for s in due_sessions():
                try:
                    await start_call(lk, s)
                except Exception:
                    logger.exception("Call start failed for session %s", s["id"])
        except Exception:
            logger.exception("Worker loop error")
        await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())
