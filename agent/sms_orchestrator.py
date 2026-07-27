"""
SMS orchestrator — fills SMS-sourced shift requests by texting nurses in ranked
order (offer -> alert call -> wait for reply), then gates on Paul's approval
before telling the facility. Runs ALONGSIDE the voice orchestrator on its own
poll loop, claiming only source='sms' rows (via claim_next_sms_request) so the
two never race on the same shift_requests row.

sms_webhook.py routes nurse YES/NO replies into sms_nurse_offers and Paul's OK into
sms_shift_state — that's what this orchestrator polls for.

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


def handle_request(req: dict):
    fac = req["facilities"]
    logger.info("SMS-filling request %s: %s %s %s at %s",
                req["id"], req["role"], req["shift_type"], req["date"], fac["name"])

    pool = db.get_candidate_pool(fac["slug"], req["date"], req["shift_type"], req["role"])
    if not pool:
        logger.info("No eligible nurses for request %s", req["id"])
        db.mark_request_unfilled(req["id"])
        db.send_sms(req["facility_callback_number"],
                    f"Sorry, no one was available for the {req['shift_type'].lower()} "
                    f"shift on {db.pretty_date(req['date'])} yet. We'll keep trying.")
        if db.DEV:
            db.mark_request_done_dev(req["id"])
        return

    ranked, reason = rank_pool(pool, req)
    ranked = rotate_top10(ranked)
    db.create_sms_offers(req["id"], ranked)

    while True:
        offer = db.get_next_pending_offer(req["id"])
        if not offer:
            logger.info("SMS offer pool exhausted for request %s", req["id"])
            db.mark_request_unfilled(req["id"])
            db.mark_sms_state(req["id"], "no_availability")
            db.send_sms(req["facility_callback_number"],
                        f"Sorry, no one was available for the {req['shift_type'].lower()} "
                        f"shift on {db.pretty_date(req['date'])} yet. We'll keep trying.")
            if ADMIN_PHONE:
                db.send_sms(ADMIN_PHONE, f"No one accepted request {req['id']} "
                            f"({fac['name']}, {db.pretty_date(req['date'])} "
                            f"{req['shift_type']}).")
            if db.DEV:
                db.mark_request_done_dev(req["id"])
            return

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
        # declined or no_reply -> loop continues to the next pending offer


def main():
    logger.info("SMS orchestrator running. Polling every %ss.", POLL_SECONDS)
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
