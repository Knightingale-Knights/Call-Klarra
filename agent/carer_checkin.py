"""
Inactive carer check-ins.

Daily at 10am Melbourne, Klarra texts up to CHECKIN_DAILY_LIMIT (15) carers who have:
  - no shift in the last 30 days, and
  - no availability in the next 30 days,
  - are still active, and haven't been checked in with in the last 30 days
(Supabase RPC carers_due_checkin, longest idle first). Each carer is re-checked against
Bubble before texting, so anyone Paul has already deactivated there is skipped.

Replies (routed here by sms_webhook while a check-in is open, 7 days):
  YES -> asked to add availability in the app; Paul gets an FYI.
  NO  -> marked inactive in Bubble (User 'account status' = no and 'active' = no) and in Supabase
         (nurses.active = false); Paul gets an FYI.
  No reply within 24h -> treated as NO: marked inactive (no text to the carer), Paul
         gets one summary text. A YES within 7 days of the check-in still reactivates them.
  Anything else goes to the help desk as normal.

Every check-in text and reply is also logged in helpdesk_messages, so Klarra has the
context if the carer keeps texting.

Table: carer_checkins (nurse_id, phone, message, status sent|yes|no|no_reply|failed, reply,
sent_at, replied_at). Started by sms_orchestrator.py (start_worker).
"""

import os
import time
import logging
import threading
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

import db

logger = logging.getLogger("knightingale-checkin")

MELB = ZoneInfo("Australia/Melbourne")
ADMIN_PHONE = "+61426512584"
SEND_HOUR = 10
DAILY_LIMIT = int(os.environ.get("CHECKIN_DAILY_LIMIT", "15"))
REPLY_WINDOW_DAYS = 7
POLL_SECONDS = 300
# First day texts may go out (YYYY-MM-DD, Melbourne).
START_DATE = os.environ.get("CHECKIN_START_DATE", "2026-10-22")
BUBBLE_BASE = "https://knightingale.com.au/api/1.1/obj"

YES = {"yes", "y", "yep", "yeah", "yes please", "yup", "sure", "definitely", "absolutely",
       "of course", "yes i would", "yes i do", "i would", "still keen", "keen"}
NO = {"no", "n", "nope", "nah", "no thanks", "no thank you", "not anymore", "no longer",
      "not interested", "no sorry"}


def _bubble_headers() -> dict:
    return {"Authorization": f"Bearer {os.environ['BUBBLE_API_TOKEN']}"}


def _message(first_name: str) -> str:
    name = (first_name or "").strip().split(" ")[0] or "there"
    return (f"Hi {name}, it's Klarra from Knightingale. Nice to meet you. We haven't seen "
            "you on a shift for a while and I can't see any availability for you? Taking "
            "the opportunity to check if you'd still like to work with us. Please let me "
            "know when you have a moment by replying YES or NO.")


def _norm(text: str) -> str:
    return " ".join((text or "").lower().replace("!", " ").replace(".", " ")
                    .replace(",", " ").split())


def _log(phone, direction, body, nurse_id=None, source="checkin"):
    try:
        db.get_client().table("helpdesk_messages").insert({
            "phone": db.normalise_phone(phone), "nurse_id": nurse_id,
            "direction": direction, "body": body, "source": source,
        }).execute()
    except Exception:
        logger.exception("Could not log check-in message")


def _text_paul(body: str) -> None:
    try:
        db.send_sms(ADMIN_PHONE, body)
    except Exception:
        logger.exception("Could not text Paul about a check-in")


def _still_active_in_bubble(bubble_user_id: str | None) -> bool:
    if not bubble_user_id:
        return False
    try:
        r = requests.get(f"{BUBBLE_BASE}/user/{bubble_user_id}", headers=_bubble_headers(),
                         timeout=20)
        if r.status_code == 404:
            return False
        r.raise_for_status()
        u = r.json().get("response", {})
        if u.get("account status") in (False, "no", "No"):
            return False
        return u.get("active") is True
    except Exception:
        logger.exception("Could not check Bubble user %s", bubble_user_id)
        return False


# --- sending -----------------------------------------------------------------

def sent_today() -> int:
    start = datetime.now(MELB).replace(hour=0, minute=0, second=0, microsecond=0)
    r = (db.get_client().table("carer_checkins").select("id", count="exact")
         .gte("sent_at", start.astimezone(timezone.utc).isoformat()).execute())
    return r.count or 0


def run_once(limit: int | None = None) -> int:
    """Send today's batch. Returns how many were texted."""
    if db.DEV or db.MID:
        logger.info("Check-ins skipped (KLARRA_MODE=%s)", db.KLARRA_MODE)
        return 0
    room = (limit if limit is not None else DAILY_LIMIT) - sent_today()
    if room <= 0:
        return 0
    client = db.get_client()
    # Ask for extra, since some will be skipped as already inactive in Bubble.
    due = client.rpc("carers_due_checkin", {"p_limit": room * 2 + 5}).execute().data or []
    skip = db.dev_testers() | {db.normalise_phone(ADMIN_PHONE)}
    sent = 0
    for n in due:
        if sent >= room:
            break
        phone = db.normalise_phone(n.get("phone"))
        if not phone or phone in skip:
            continue
        if not _still_active_in_bubble(n.get("bubble_user_id")):
            client.table("nurses").update({"active": False}).eq("id", n["id"]).execute()
            logger.info("Check-in skipped: %s is inactive in Bubble", n.get("first_name"))
            continue
        msg = _message(n.get("first_name"))
        row = client.table("carer_checkins").insert({
            "nurse_id": n["id"], "phone": phone, "message": msg,
        }).execute().data[0]
        try:
            db.send_sms(phone, msg)
            _log(phone, "out", msg, nurse_id=n["id"])
            sent += 1
        except Exception:
            logger.exception("Check-in text to %s failed", n.get("first_name"))
            client.table("carer_checkins").update({"status": "failed"}).eq("id", row["id"]).execute()
        time.sleep(1)
    if sent:
        logger.info("Sent %d carer check-ins", sent)
    return sent


NO_REPLY_HOURS = 24


def expire_no_replies() -> int:
    """Check-ins with no reply after NO_REPLY_HOURS count as a NO: the carer is marked
    inactive in Bubble and Supabase (no text to the carer), and Paul gets one summary
    text. A YES that arrives later still reactivates them (see handle_reply)."""
    if db.DEV or db.MID:
        return 0
    client = db.get_client()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=NO_REPLY_HOURS)).isoformat()
    rows = (client.table("carer_checkins")
            .select("*, nurses(id, first_name, last_name, bubble_user_id)")
            .eq("status", "sent").lt("sent_at", cutoff).limit(50).execute().data or [])
    names, failed = [], []
    for c in rows:
        claimed = (client.table("carer_checkins").update({"status": "no_reply"})
                   .eq("id", c["id"]).eq("status", "sent").execute().data or [])
        if not claimed:
            continue
        nurse = c.get("nurses") or {}
        name = f"{nurse.get('first_name') or ''} {nurse.get('last_name') or ''}".strip() or "A carer"
        if not _deactivate_in_bubble(nurse.get("bubble_user_id")):
            failed.append(name)
        client.table("nurses").update({"active": False}).eq("id", c["nurse_id"]).execute()
        names.append(name)
    if names:
        msg = (f"Check-in: no reply after {NO_REPLY_HOURS}h, so I've marked "
               f"{len(names)} inactive: {', '.join(names)}.")
        if failed:
            msg += f" Couldn't update Bubble for {', '.join(failed)}, please do those by hand."
        _text_paul(msg)
    return len(names)


def _loop() -> None:
    last_day = None
    while True:
        try:
            now = datetime.now(MELB)
            if (now.hour >= SEND_HOUR and now.date() != last_day
                    and now.date().isoformat() >= START_DATE):
                run_once()
                last_day = now.date()
        except Exception:
            logger.exception("Check-in run failed")
        try:
            expire_no_replies()
        except Exception:
            logger.exception("Check-in expiry failed")
        time.sleep(POLL_SECONDS)


def start_worker() -> None:
    threading.Thread(target=_loop, name="carer-checkins", daemon=True).start()


def _reactivate_in_bubble(bubble_user_id: str | None) -> bool:
    if not bubble_user_id:
        return False
    url = f"{BUBBLE_BASE}/user/{bubble_user_id}"
    for status_value in (True, "yes"):
        try:
            r = requests.patch(url, headers=_bubble_headers(), timeout=20,
                               json={"active": True, "account status": status_value})
            if r.ok:
                return True
            logger.warning("Bubble reactivate attempt failed (%s): %s", r.status_code, r.text)
        except Exception:
            logger.exception("Bubble reactivate request failed")
    return False


def _deactivate_in_bubble(bubble_user_id: str | None) -> bool:
    """Set the User's 'account status' and 'active' to no. Tries yes/no (boolean)
    values first, then the text 'no', in case account status is a text field."""
    if not bubble_user_id:
        return False
    url = f"{BUBBLE_BASE}/user/{bubble_user_id}"
    for status_value in (False, "no"):
        try:
            r = requests.patch(url, headers=_bubble_headers(), timeout=20,
                               json={"active": False, "account status": status_value})
            if r.ok:
                return True
            logger.warning("Bubble deactivate attempt failed (%s): %s", r.status_code, r.text)
        except Exception:
            logger.exception("Bubble deactivate request failed")
    return False


# --- replies -----------------------------------------------------------------

def open_checkin(phone: str) -> dict | None:
    since = (datetime.now(timezone.utc) - timedelta(days=REPLY_WINDOW_DAYS)).isoformat()
    r = (db.get_client().table("carer_checkins")
         .select("*, nurses(id, first_name, last_name, bubble_user_id)")
         .eq("phone", db.normalise_phone(phone)).in_("status", ["sent", "no_reply"])
         .gte("sent_at", since).order("sent_at", desc=True).limit(1).execute())
    return r.data[0] if r.data else None


def handle_reply(phone: str, body: str) -> str | None:
    """A carer's YES/NO to an open check-in. Returns the reply text, or None if there is
    no open check-in or the text isn't a clear yes/no (caller carries on)."""
    checkin = open_checkin(phone)
    if not checkin:
        return None
    t = _norm(body)
    answer = "yes" if t in YES else "no" if t in NO else None
    if not answer:
        return None
    late = checkin["status"] == "no_reply"
    if late and answer == "no":
        # Already treated as a NO: just record what they said.
        db.get_client().table("carer_checkins").update({
            "status": "no", "reply": body,
            "replied_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", checkin["id"]).execute()

    client = db.get_client()
    nurse = checkin.get("nurses") or {}
    name = f"{nurse.get('first_name') or ''} {nurse.get('last_name') or ''}".strip() or "A carer"
    _log(phone, "in", body, nurse_id=checkin["nurse_id"])
    client.table("carer_checkins").update({
        "status": answer, "reply": body,
        "replied_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", checkin["id"]).execute()

    if answer == "yes":
        reply = ("Oh that's great news. Thanks so much. When you have a sec, can you update "
                 "your availability please?")
        if late:
            # They were marked inactive for not replying in time: switch them back on.
            ok = _reactivate_in_bubble(nurse.get("bubble_user_id"))
            client.table("nurses").update({"active": True}).eq("id", checkin["nurse_id"]).execute()
            _text_paul(f"Check-in: {name} replied YES after the 24h cutoff, so I've made them "
                       "active again and asked them to add availability."
                       + ("" if ok else " Couldn't update Bubble, please set them active there."))
        else:
            _text_paul(f"Check-in: {name} still wants to work. I've asked them to add availability.")
    elif late:
        reply = ("Oh, that's such a shame. We hate to see you go. If you ever want to change "
                 "your mind, please let us know. We now work across Wintringham, Brotherhood "
                 "of St Laurence and VMCH so we have plenty of shifts to keep you busy =)")
    else:
        problem = "" if _deactivate_in_bubble(nurse.get("bubble_user_id")) else (
            " Couldn't update Bubble, please set account status and active to no there.")
        client.table("nurses").update({"active": False}).eq("id", checkin["nurse_id"]).execute()
        reply = ("Oh, that's such a shame. We hate to see you go. If you ever want to change "
                 "your mind, please let us know. We now work across Wintringham, Brotherhood "
                 "of St Laurence and VMCH so we have plenty of shifts to keep you busy =)")
        _text_paul(f"Check-in: {name} replied NO, so I've marked them inactive.{problem}")

    _log(phone, "out", reply, nurse_id=checkin["nurse_id"])
    return reply
