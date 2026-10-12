"""
Questionnaire worker (Railway service, always on). Everything happens by SMS.

Every minute:
  1. Heads-up (5pm-9pm Melbourne, every 15 min, idempotent):
     for each NDIS participant with a shift tomorrow (first one starting 7am-7pm) and
     empty questionnaire fields, create a Questionnaire Session for that shift and text
     the carer a heads-up naming what's missing and when they'll be texted.
  2. Start: for each Scheduled session whose time has passed, check the shift is still on
     and the carer has checked in (Bubble "check in time" set), then text the first
     question (questionnaire_sms.start). Not checked in: retry every 5 min until 90 min
     after shift start, then mark Failed.
  3. Timeouts: remind once after 45 min of silence, close after 90 min.
  4. Risk plans: after a Risk Assessment or Home Safety session ends, create a Risk
     Management Plan in Bubble for each newly identified risk (risk_plans.py).

The carer's replies come in through sms_webhook.py, which hands them to
questionnaire_sms.handle_inbound.

Env:
  QUESTIONNAIRE_ONLY_PARTICIPANT  optional Bubble user id: restrict to one participant (testing)
  QUESTIONNAIRE_TEST_NOW=1        on start, schedule a test in 2 minutes (and skip the 5pm heads-up run)
  QUESTIONNAIRE_TEST_SHIFT        Bubble shift id to test with (else ONLY's next shift)
  QUESTIONNAIRE_SKIP_CHECKIN=1    don't wait for the carer's check in (testing)
  QUESTIONNAIRE_TEST_SECTION      force a section in test mode (e.g. Home Safety)
  QUESTIONNAIRE_BUBBLE_BASE       Bubble Data API base (dev: https://knightingale.com.au/version-test/api/1.1/obj)
  plus the usual BUBBLE_API_TOKEN, SUPABASE_*, TWILIO_*, ANTHROPIC_API_KEY, KLARRA_MODE.

Run:  python agent/questionnaire_worker.py
"""

import os
import time
import logging
from datetime import datetime, timedelta, timezone

import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from dotenv import load_dotenv

load_dotenv()

import db
import questionnaire as q
import questionnaire_sms as qs
import risk_plans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("questionnaire-worker")
for noisy in ("hpack", "httpx", "httpcore", "h2"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

TEST_NOW = os.environ.get("QUESTIONNAIRE_TEST_NOW") == "1"
SKIP_CHECKIN = os.environ.get("QUESTIONNAIRE_SKIP_CHECKIN") == "1"
ONLY = os.environ.get("QUESTIONNAIRE_ONLY_PARTICIPANT", "").strip() or None
TEST_SHIFT = os.environ.get("QUESTIONNAIRE_TEST_SHIFT", "").strip() or None
TEST_SECTION = os.environ.get("QUESTIONNAIRE_TEST_SECTION", "").strip() or None

HEADS_UP_START, HEADS_UP_END = 17, 21      # Melbourne hours
EARLIEST_START, LATEST_START = 7, 19        # only shifts starting in this window
START_OFFSET = timedelta(minutes=30)
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


def schedule_one(participant_bid: str, shift: dict, start_at: datetime | None = None,
                 when: str = "tomorrow", force: bool = False) -> None:
    if not force and session_exists_for_shift(shift["_id"]):
        logger.info("Session already exists for shift %s", shift["_id"])
        return
    if force and TEST_SECTION in q.SECTIONS:
        rec = q.get_record(TEST_SECTION, participant_bid)
        nxt = (TEST_SECTION, q.missing_fields(TEST_SECTION, rec))
    else:
        nxt = q.next_section(participant_bid)
    if not nxt or not nxt[1]:
        logger.info("Nothing left to ask for participant %s", participant_bid)
        return
    section, askable = nxt
    participant = q.bubble_user(participant_bid)
    carer = q.bubble_user(shift["carer"])
    phone = db.normalise_phone(carer.get("phone number"))
    if not phone:
        logger.warning("Carer %s has no phone, skipping", shift["carer"])
        return
    start_at = start_at or (q.shift_start(shift) + START_OFFSET)
    p_name = q.first_name(participant)
    pr = q.pronouns(participant)
    now = datetime.now(timezone.utc).isoformat()

    bubble_id = q.bubble_create("questionnairesession", {
        "participant": participant_bid, "carer": shift["carer"], "shift": shift["_id"],
        "section": section, "status": "Scheduled",
        "scheduled call": start_at.isoformat(), "heads up sent": now,
    })
    db.get_client().table("questionnaire_sessions").insert({
        "bubble_id": bubble_id, "participant_bubble_id": participant_bid,
        "carer_bubble_id": shift["carer"], "carer_phone": phone, "section": section,
        "status": "Scheduled", "shift_bubble_id": shift["_id"],
        "scheduled_call": start_at.isoformat(), "heads_up_sent": now,
    }).execute()

    topics = q.topics_phrase(section, askable).replace("their ", f"{pr['poss']} ")
    body = (f"Hi {q.first_name(carer)}, I noticed we don't have "
            f"any information for {p_name} on {topics}. I'll text you {when} at "
            f"{q.pretty_time(start_at)} when you're with {pr['obj']}, so you can ask "
            f"{pr['obj']} a few quick questions for me. Thanks!")
    db.send_sms(phone, body)
    logger.info("Heads-up sent: %s / %s / %s at %s", p_name, section, phone, start_at)


def run_heads_up() -> None:
    shifts = tomorrows_participant_shifts()
    logger.info("Heads-up pass: %d participants with shifts tomorrow", len(shifts))
    for pbid, shift in shifts.items():
        try:
            schedule_one(pbid, shift)
        except Exception:
            logger.exception("Heads-up failed for participant %s", pbid)


# ---------- 2. start ----------

def due_sessions() -> list[dict]:
    now = datetime.now(timezone.utc).isoformat()
    r = (db.get_client().table("questionnaire_sessions").select("*")
         .eq("status", "Scheduled").lte("scheduled_call", now).execute())
    return r.data or []


def start_session(session: dict) -> None:
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

    # Carer may have changed since the heads-up: text whoever is on the shift now.
    carer_bid = shift.get("carer") or session["carer_bubble_id"]
    carer = q.bubble_user(carer_bid)
    participant = q.bubble_user(session["participant_bubble_id"])
    phone = qs.effective_number(carer.get("phone number") or session.get("carer_phone"))
    if not phone:
        q.update_session(session, status="Failed", skipped_fields="blocked by KLARRA_MODE")
        return
    if carer_bid != session["carer_bubble_id"]:
        q.update_session(session, carer_bubble_id=carer_bid)
    qs.start(session, carer, participant, phone)


def schedule_test_now() -> None:
    """Testing: schedule TEST_SHIFT (or ONLY's next shift) to start in 2 minutes."""
    global ONLY
    if TEST_SHIFT:
        shift = q.bubble_get_by_id("shift", TEST_SHIFT)
        if not shift or not shift.get("participant") or not shift.get("carer"):
            logger.warning("Test shift %s not found, or has no participant/carer", TEST_SHIFT)
            return
        ONLY = shift["participant"]
    else:
        if not ONLY:
            logger.warning("QUESTIONNAIRE_TEST_NOW needs QUESTIONNAIRE_TEST_SHIFT or ONLY_PARTICIPANT")
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
    # Close any earlier test session still open, then start fresh every restart.
    old = (db.get_client().table("questionnaire_sessions").select("id")
           .eq("shift_bubble_id", shift["_id"]).in_("status", ["Scheduled", "In Progress"])
           .execute())
    for row in old.data or []:
        db.get_client().table("questionnaire_sessions").update(
            {"status": "Failed", "skipped_fields": "replaced by new test"}).eq("id", row["id"]).execute()
    logger.info("Test shift %s, participant %s", shift["_id"], ONLY)
    schedule_one(ONLY, shift, start_at=q.melb_now() + timedelta(minutes=2), when="today",
                 force=True)


def main():
    last_heads_up = 0.0
    logger.info("Questionnaire worker started (KLARRA_MODE=%s, only=%s, bubble=%s)",
                db.KLARRA_MODE, ONLY, q.BUBBLE_BASE)
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
                    start_session(s)
                except Exception:
                    logger.exception("Start failed for session %s", s["id"])
            qs.check_timeouts()
            risk_plans.process_pending()
        except Exception:
            logger.exception("Worker loop error")
        time.sleep(60)


if __name__ == "__main__":
    main()
