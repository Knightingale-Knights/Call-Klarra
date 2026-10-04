"""
Duplicate shift detection — finds any two shifts for the same participant, on the
same date, with the same start and end time, and queues them for Paul to resolve
by SMS.

Only one review is ever texted at a time ('sent'); a bare '1' or '2' reply from
Paul's number is unambiguous because of this. Resolving one (see sms_webhook.py's
handle_duplicate_review_reply) immediately sends the next queued pair, so a whole
backlog clears as a quick back-and-forth rather than one text per cron run.

This script only detects and queues/sends — it never deletes anything itself.
Run on demand (e.g. right after the weekly generator, or manually against a
backlog):  python agent/duplicate_shift_review.py
"""

import os
import logging

import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from dotenv import load_dotenv

import db

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("duplicate-shift-review")

ADMIN_PHONE = "+61426512584"


def describe_side(shift: dict) -> str:
    nurse = db.get_nurse(shift["nurse_id"])
    name = (nurse or {}).get("first_name") or f"nurse {shift['nurse_id']}"
    recurring = "recurring" if shift.get("recurring_template_id") else "one-off"
    return f"{name} ({recurring})"


def hhmm_from_timestamp(ts: str) -> str:
    """Extract 'HH:MM' from a timestamptz string, regardless of whether it uses a
    'T' or a space separator (Postgres/PostgREST can return either)."""
    time_part = ts.split("T")[-1] if "T" in ts else ts.split(" ")[-1]
    return time_part[:5]


def send_review_sms(review: dict) -> None:
    shift_1 = db.get_shift(review["shift_1_id"])
    shift_2 = db.get_shift(review["shift_2_id"])
    if not (shift_1 and shift_2):
        logger.error("Review %s: missing shift row(s), marking resolved with no action",
                     review["id"])
        db.resolve_duplicate_review(review["id"], review["shift_1_id"], review["shift_2_id"])
        return
    participant = db.get_participant(shift_1["participant_id"])
    pname = (participant or {}).get("name") or f"participant {shift_1['participant_id']}"
    body = (
        f"Duplicate shift for {pname} on {db.pretty_date(shift_1['date'])}, "
        f"{hhmm_from_timestamp(shift_1['start_time'])}-"
        f"{hhmm_from_timestamp(shift_1['end_time'])}.\n"
        f"1: {describe_side(shift_1)}\n"
        f"2: {describe_side(shift_2)}\n"
        f"Reply 1 or 2 to keep that one."
    )
    db.send_sms(ADMIN_PHONE, body)
    db.mark_duplicate_review_sent(review["id"])
    logger.info("Sent duplicate review %s", review["id"])


def queue_new_duplicates() -> int:
    pairs = db.find_duplicate_shift_pairs()
    already = db.already_reviewed_shift_ids()
    queued = 0
    for shift_1, shift_2 in pairs:
        if shift_1["id"] in already or shift_2["id"] in already:
            continue
        db.create_duplicate_review(shift_1["id"], shift_2["id"])
        already.add(shift_1["id"])
        already.add(shift_2["id"])
        queued += 1
    return queued


def main():
    queued = queue_new_duplicates()
    logger.info("Queued %d new duplicate pair(s)", queued)

    if db.get_sent_duplicate_review():
        logger.info("A review is already awaiting Paul's reply — not sending another")
        return

    next_review = db.get_next_pending_duplicate_review()
    if next_review:
        send_review_sms(next_review)
    else:
        logger.info("No pending duplicates to send")


if __name__ == "__main__":
    main()
