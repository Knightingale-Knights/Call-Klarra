"""
SMS orchestrator — fills SMS-sourced shift requests by texting nurses in ranked
order (offer -> alert call -> wait for reply), then gates on Paul's approval
before telling the facility. Runs ALONGSIDE the voice orchestrator on its own
poll loop, claiming only source='sms' rows (via claim_next_sms_request) so the
two never race on the same shift_requests row.

sms_webhook.py routes nurse YES/NO replies into sms_nurse_offers and Paul's OK into
sms_shift_state — that's what this orchestrator polls for.

Role escalation: if nobody at the requested level takes the shift, the loop moves up
db.escalation_ladder() (PCA -> EN -> RN) and appends that tier's candidates below the
exhausted ones, texting the admins each time it steps up. A higher tier usually bills
at a higher rate, so escalation only happens after every candidate at the current level
has declined or not replied.

KLARRA_MODE=mid changes the whole shape of this: the nurse cascade is skipped
entirely. The dev phones get the full request immediately so Paul/Vidhu can fill it
by hand, and 30 seconds later the facility is texted the name of the top-ranked
candidate — so the facility experiences a fast automatic fill while a human is
actually doing the work behind it. See handle_request_mid().

NOTE ON MID: the name goes to the facility at 30s whether or not that nurse has
agreed to anything. If they later decline, the facility has already been told they're
covering it and someone has to walk that back manually.

Run:  python agent/sms_orchestrator.py
"""

import os
import time
import logging

import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from dotenv import load_dotenv

import db
from orchestrator import rank_pool, rotate_top10

load_dotenv()
logger = logging.getLogger("knightingale-sms-orchestrator")
logging.basicConfig(level=logging.INFO)

POLL_SECONDS = 5
ALERT_DELAY_SECONDS = 10       # wait after offer text before the alert call
REPLY_TIMEOUT_SECONDS = 30     # wait after the alert call before giving up on this nurse
REPLY_POLL_SECONDS = 3

ADMIN_PHONE = os.environ.get("KLARRA_DEV_PHONE")
REMINDER_INTERVAL_SECONDS = 300  # 5 min
MAX_REMINDERS = 2

MID_FACILITY_DELAY_SECONDS = 30  # mid: pause before naming the nurse to the facility


def _offer_status(offer_id: str) -> str | None:
    client = db.get_client()
    r = (client.table("sms_nurse_offers").select("status")
         .eq("id", offer_id).limit(1).execute())
    return r.data[0]["status"] if r.data else None


def _short_date(d: str) -> str:
    """Format 'YYYY-MM-DD' as 'Tue 28th'. Returns input unchanged on failure."""
    from datetime import datetime
    try:
        dt = datetime.strptime(str(d)[:10], "%Y-%m-%d")
        day = dt.day
        suffix = "th" if 11 <= day % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        return dt.strftime("%a ") + f"{day}{suffix}"
    except Exception:
        return str(d)


def _compact_time(t: str) -> str:
    """'07:00' -> '0700'."""
    return t.replace(":", "") if t else t


def _shift_time_desc(req: dict) -> str:
    if req.get("start_time") and req.get("end_time"):
        return f"{_compact_time(req['start_time'])} - {_compact_time(req['end_time'])}"
    return req["shift_type"]


def offer_message(nurse: dict, req: dict) -> str:
    nice_date = _short_date(req["date"])
    if req.get("start_time") and req.get("end_time"):
        time_desc = f"from {_compact_time(req['start_time'])} - {_compact_time(req['end_time'])}"
    else:
        time_desc = f"({req['shift_type']})"
    return (
        f"Hi {nurse['first_name']}, I've got a shift at {req['facilities']['name']} "
        f"on {nice_date} {time_desc}. Please reply YES if you would like it. "
        f"Please reply NO if you would prefer to pass. Thank you"
    )


def text_admins(body: str) -> None:
    """Text every dev/admin number (KLARRA_DEV_PHONES + KLARRA_DEV_PHONE)."""
    for phone in db.dev_testers():
        try:
            db.send_sms(phone, body)
        except Exception:
            logger.exception("Failed to text admin %s", phone)


def text_offer_outcome(nurse: dict, req: dict, outcome: str) -> None:
    """Text Paul each time a nurse's SMS offer resolves (accepted/declined/no_reply) —
    mirrors outbound.py's text_outcome for voice calls."""
    if not ADMIN_PHONE:
        return
    body = (
        f"Nurse text: {outcome}\n"
        f"Nurse: {nurse['first_name']}\n"
        f"Facility: {req['facilities']['name']}\n"
        f"Date: {db.pretty_date(req['date'])}\n"
        f"Shift: {req['shift_type']}"
    )
    try:
        db.send_sms(ADMIN_PHONE, body)
    except Exception:
        logger.exception("Failed to send offer outcome SMS")


def send_fyi(req: dict, nurse: dict, ranked: list[dict]) -> None:
    """Text Paul the final ranked list with the winner checked off — mirrors
    orchestrator.py's send_fyi for voice."""
    if not ADMIN_PHONE:
        return
    start = req.get("start_time") or ""
    end = req.get("end_time") or ""
    shift_time = f"{start} - {end}" if start and end else req["shift_type"]
    body = (
        f"Shift filled (SMS)\n"
        f"Nurse: {nurse['first_name']}\n"
        f"Facility: {req['facilities']['name']}\n"
        f"Date: {db.pretty_date(req['date'])}\n"
        f"Shift: {shift_time}\n"
        f"Nurses:\n"
    )
    for i, n in enumerate(ranked[:10], 1):
        marker = "✓ " if n["nurse_id"] == nurse["id"] else ""
        body += f"{i} - {marker}{n['first_name']}: {int(n.get('reliability') or 0)}\n"
    try:
        db.send_sms(ADMIN_PHONE, body.strip())
    except Exception:
        logger.exception("Failed to send SMS FYI")


def notify_escalation(req: dict, new_role: str) -> None:
    """Tell the admins the shift has moved up a rung. Escalation normally means a
    higher billing rate than the facility asked for, so it should never happen
    silently."""
    body = (
        f"Escalating shift — no {req['role']} took it.\n"
        f"Now offering to: {new_role}\n"
        f"Facility: {req['facilities']['name']}\n"
        f"Date: {db.pretty_date(req['date'])}\n"
        f"Shift: {_shift_time_desc(req)}"
    )
    text_admins(body)


def offer_nurse(offer: dict, req: dict) -> str:
    """Text one nurse the offer, alert-call them, wait for a reply. Returns the
    resulting offer status: accepted, declined, or no_reply. Texts Paul the outcome
    as soon as it's known."""
    nurse = offer["nurses"]
    logger.info("Offering request %s to %s (rank %s)", req["id"], nurse["first_name"],
                offer["rank_position"])

    db.send_sms(nurse["phone"], offer_message(nurse, req))
    db.mark_offer(offer["id"], "offered", offered_at="now()")

    waited = 0
    while waited < ALERT_DELAY_SECONDS:
        time.sleep(REPLY_POLL_SECONDS)
        waited += REPLY_POLL_SECONDS
        status = _offer_status(offer["id"])
        if status in ("accepted", "declined"):
            text_offer_outcome(nurse, req, status)
            return status

    db.place_alert_call(nurse["phone"])
    db.mark_offer(offer["id"], "alerted", alert_called_at="now()")

    waited = 0
    while waited < REPLY_TIMEOUT_SECONDS:
        time.sleep(REPLY_POLL_SECONDS)
        waited += REPLY_POLL_SECONDS
        status = _offer_status(offer["id"])
        if status in ("accepted", "declined"):
            text_offer_outcome(nurse, req, status)
            return status

    db.mark_offer(offer["id"], "no_reply")
    text_offer_outcome(nurse, req, "no_reply")
    return "no_reply"


def request_admin_approval(req: dict, nurse: dict) -> bool:
    """Text Paul, wait for OK, sending up to 2 reminders 5 min apart. After the
    2nd reminder with no OK, auto-confirm (per spec) and return True. Always
    returns True currently — kept as a function so the gate lives in one place."""
    body = (
        f"Nurse confirmed for shift.\n"
        f"Nurse: {nurse['first_name']}\n"
        f"Facility: {req['facilities']['name']}\n"
        f"Date: {db.pretty_date(req['date'])}\n"
        f"Shift: {req['shift_type']}\n\n"
        f"Reply OK to approve."
    )
    db.send_sms(ADMIN_PHONE, body)
    db.mark_sms_state(req["id"], "pending_admin_approval", admin_notified_at="now()")

    reminders_sent = 0
    while True:
        waited = 0
        while waited < REMINDER_INTERVAL_SECONDS:
            time.sleep(REPLY_POLL_SECONDS)
            waited += REPLY_POLL_SECONDS
            state = db.get_sms_state(req["id"])
            if state and state["status"] == "confirmed":
                return True
        if reminders_sent >= MAX_REMINDERS:
            logger.info("No admin OK after %d reminders — auto-confirming request %s",
                        MAX_REMINDERS, req["id"])
            return True
        reminders_sent += 1
        db.send_sms(ADMIN_PHONE, f"Reminder {reminders_sent}/{MAX_REMINDERS}: " + body)
        db.mark_sms_state(req["id"], "pending_admin_approval",
                          admin_reminder_count=reminders_sent)


# --- MID MODE -------------------------------------------------------------

def handle_request_mid(req: dict):
    """
    Mid mode: no nurse cascade at all.

      1. Rank the real candidate pool (read-only — nobody is contacted).
      2. Text every dev phone the full request immediately, with the ranked
         shortlist, so Paul/Vidhu can go and fill it by hand.
      3. Wait 30s, then text the facility naming the TOP-RANKED candidate —
         regardless of whether that nurse has agreed to anything.
      4. Park the row as 'mid_handoff' so it isn't re-claimed. The actual booking
         is recorded manually in Bubble.

    db.send_sms blocks any number that isn't a dev tester or a registered facility
    number while KLARRA_MODE=mid, so a nurse cannot be contacted from this path.
    """
    fac = req["facilities"]
    logger.info("[MID] handling request %s: %s %s %s at %s",
                req["id"], req["role"], req["shift_type"], req["date"], fac["name"])

    pool = db.get_candidate_pool(fac["slug"], req["date"], req["shift_type"], req["role"])
    ranked: list[dict] = []
    if pool:
        ranked, _reason = rank_pool(pool, req)
        ranked = rotate_top10(ranked)

    # 1. Admins get the request straight away.
    header = (
        f"MID — shift request in, fill this one manually.\n"
        f"Facility: {fac['name']}\n"
        f"Date: {db.pretty_date(req['date'])}\n"
        f"Shift: {_shift_time_desc(req)}\n"
        f"Role: {req['role']}\n"
        f"From: {req['facility_callback_number']}\n"
    )
    if ranked:
        header += "Ranked:\n"
        for i, n in enumerate(ranked[:10], 1):
            header += f"{i} - {n['first_name']}: {int(n.get('reliability') or 0)}\n"
        header += (
            f"\nFacility will be told {ranked[0]['first_name']} is covering it "
            f"in {MID_FACILITY_DELAY_SECONDS}s."
        )
    else:
        header += "\nNo eligible candidates — facility gets a holding message."
    text_admins(header.strip())

    # 2. Pause, then tell the facility.
    time.sleep(MID_FACILITY_DELAY_SECONDS)

    callback = req.get("facility_callback_number")
    if ranked and callback:
        top = ranked[0]
        db.send_sms(
            callback,
            f"Good news — {top['first_name']} is covering the "
            f"{req['shift_type'].lower()} shift on {db.pretty_date(req['date'])}."
        )
        logger.info("[MID] told %s that %s is covering request %s",
                    callback, top["first_name"], req["id"])
    elif callback:
        db.send_sms(
            callback,
            f"Thanks — we're confirming someone for the {req['shift_type'].lower()} "
            f"shift on {db.pretty_date(req['date'])} now and will come back to you shortly."
        )
        text_admins(f"MID — no candidates for request {req['id']} ({fac['name']}, "
                    f"{db.pretty_date(req['date'])} {req['shift_type']}). "
                    f"Facility got a holding message.")

    # 3. Park it — filled by hand in Bubble, not by this loop.
    db.mark_request_status(req["id"], "mid_handoff")


# --- LIVE / DEV MODE ------------------------------------------------------

def handle_request(req: dict):
    if db.MID:
        return handle_request_mid(req)

    fac = req["facilities"]
    logger.info("SMS-filling request %s: %s %s %s at %s",
                req["id"], req["role"], req["shift_type"], req["date"], fac["name"])

    ladder = db.escalation_ladder(req["role"])
    ranked: list[dict] = []
    tier_index = 0
    tried_any = False

    while True:
        offer = db.get_next_pending_offer(req["id"])

        if not offer:
            # Current tier is exhausted (or we haven't loaded one yet). Move up the
            # ladder until a tier actually has candidates, then keep going.
            escalated = False
            while tier_index < len(ladder):
                role = ladder[tier_index]
                tier_index += 1
                pool = db.get_candidate_pool(fac["slug"], req["date"],
                                             req["shift_type"], role)
                if not pool:
                    logger.info("No %s candidates for request %s", role, req["id"])
                    continue
                tier_ranked, _reason = rank_pool(pool, req)
                tier_ranked = rotate_top10(tier_ranked)
                db.create_sms_offers(req["id"], tier_ranked)
                ranked = ranked + [n for n in tier_ranked if n not in ranked]
                if tried_any:
                    notify_escalation(req, role)
                logger.info("Request %s now offering at %s tier (%d candidates)",
                            req["id"], role, len(tier_ranked))
                escalated = True
                break

            if escalated:
                offer = db.get_next_pending_offer(req["id"])

            if not offer:
                logger.info("All tiers exhausted for request %s", req["id"])
                db.mark_request_unfilled(req["id"])
                db.mark_sms_state(req["id"], "no_availability")
                db.send_sms(req["facility_callback_number"],
                            f"Sorry, no one was available for the {req['shift_type'].lower()} "
                            f"shift on {db.pretty_date(req['date'])} yet. We'll keep trying.")
                if ADMIN_PHONE:
                    db.send_sms(ADMIN_PHONE, f"No one accepted request {req['id']} "
                                f"({fac['name']}, {db.pretty_date(req['date'])} "
                                f"{req['shift_type']}, {req['role']} + escalations).")
                if db.DEV:
                    db.mark_request_done_dev(req["id"])
                return

        tried_any = True
        outcome = offer_nurse(offer, req)
        if outcome == "accepted":
            nurse = offer["nurses"]
            approved = request_admin_approval(req, nurse)
            if approved:
                db.assign_availability(nurse["id"], req["date"], req["shift_type"])
                db.mark_request_filled(req["id"], nurse["id"])
                db.mark_sms_state(req["id"], "confirmed", confirmed_nurse_id=nurse["id"],
                                  admin_approved_at="now()")
                db.send_sms(req["facility_callback_number"],
                            f"Good news — {nurse['first_name']} is covering the "
                            f"{req['shift_type'].lower()} shift on "
                            f"{db.pretty_date(req['date'])}.")
                send_fyi(req, nurse, ranked)
            if db.DEV:
                db.mark_request_done_dev(req["id"])
            return
        # declined or no_reply -> next pending offer, or the next tier up


def main():
    logger.info("SMS orchestrator running (KLARRA_MODE=%s). Polling every %ss.",
                db.KLARRA_MODE, POLL_SECONDS)
    while True:
        try:
            req = db.claim_next_sms_request()
            if req:
                handle_request(req)
            else:
                time.sleep(POLL_SECONDS)
        except Exception:
            logger.exception("Error handling SMS request")
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
