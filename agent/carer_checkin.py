"""
Inactive carer check-ins.

Daily at 10am Melbourne, Klarra texts up to CHECKIN_DAILY_LIMIT (15) carers who have:
  - no shift in the last 30 days, and
  - no availability in the next 30 days,
  - are still active, an account created more than 30 days ago, and haven't been checked in
    with in the last 30 days
(Supabase RPC carers_due_checkin, longest idle first). Each carer is re-checked against
Bubble before texting: anyone Paul has already deactivated there is skipped, and so is
anyone with a shift in Bubble in the last 30 days (Supabase only holds facility shifts).

Follow-up: 2 days after a check-in, carers who replied YES are checked for availability
in Bubble. If they have some, nothing is sent; if not, one follow-up text (10am to 6pm).
2 days after that, if there is still no availability, account status and active are set
to no in Bubble and nurses.active=false. No further text to the carer; Paul gets an FYI.

Replies (routed here by sms_webhook while a check-in is open, 7 days):
  YES -> asked to add availability in the app; Paul gets an FYI.
  NO  -> marked inactive in Bubble (User 'account status' = no and 'active' = no) and in Supabase
         (nurses.active = false); Paul gets an FYI.
  No reply within 24h -> treated as NO: marked inactive and sent a goodbye text, Paul
         gets one summary text. A YES within 7 days of the check-in still reactivates them.
  Anything else counts as a reply (never paused) and goes to the help desk for Paul.

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


def _account_older_than(nurse_id: int, bubble_user_id: str | None, days: int = 30) -> bool:
    """True if the carer's account was created more than `days` days ago. Uses the Bubble
    user's Created Date, falling back to nurses.created_at. On any doubt, returns False
    (don't text)."""
    created = None
    try:
        if bubble_user_id:
            r = requests.get(f"{BUBBLE_BASE}/user/{bubble_user_id}", headers=_bubble_headers(),
                             timeout=20)
            if r.ok:
                created = r.json().get("response", {}).get("Created Date")
        if not created:
            row = (db.get_client().table("nurses").select("created_at").eq("id", nurse_id)
                   .limit(1).execute().data or [])
            created = row[0].get("created_at") if row else None
        if not created:
            return False
        dt = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt <= datetime.now(timezone.utc) - timedelta(days=days)
    except Exception:
        logger.exception("Could not check account age for nurse %s", nurse_id)
        return False


def _bubble_search(datatype: str, constraints: list, limit: int = 20) -> list[dict] | None:
    """Search a Bubble data type. Returns None if the request failed."""
    import json
    try:
        r = requests.get(f"{BUBBLE_BASE}/{datatype}", headers=_bubble_headers(), timeout=20,
                         params={"limit": limit, "constraints": json.dumps(constraints)})
        r.raise_for_status()
        return r.json().get("response", {}).get("results", [])
    except Exception:
        logger.exception("Bubble %s search failed", datatype)
        return None


def _worked_recently_in_bubble(bubble_user_id: str | None, days: int = 30) -> bool:
    """True if Bubble has a non-cancelled shift for this carer in the last `days` days.
    Supabase only holds facility shifts from the daily sync, so NDIS and other shifts
    can be missing there. On a failed lookup, returns True (don't text)."""
    if not bubble_user_id:
        return False
    today = datetime.now(MELB).date()
    start = (today - timedelta(days=days + 1)).isoformat()
    end = (today + timedelta(days=1)).isoformat()
    shifts = _bubble_search("shift", [
        {"key": "carer", "constraint_type": "equals", "value": bubble_user_id},
        {"key": "date", "constraint_type": "greater than", "value": start},
        {"key": "date", "constraint_type": "less than", "value": end},
    ])
    if shifts is None:
        return True
    return any(not s.get("cancelled") for s in shifts)


def _has_future_availability_in_bubble(bubble_user_id: str | None) -> bool | None:
    """True if the carer has any availability from today on in Bubble. None if the
    lookup failed."""
    if not bubble_user_id:
        return False
    yesterday = (datetime.now(MELB).date() - timedelta(days=1)).isoformat()
    avails = _bubble_search("availability", [
        {"key": "carer", "constraint_type": "equals", "value": bubble_user_id},
        {"key": "available", "constraint_type": "equals", "value": "true"},
        {"key": "date", "constraint_type": "greater than", "value": yesterday},
    ], limit=1)
    if avails is None:
        return None
    return len(avails) > 0


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
        if not _account_older_than(n["id"], n.get("bubble_user_id")):
            logger.info("Check-in skipped: %s's account is under 30 days old", n.get("first_name"))
            continue
        if _worked_recently_in_bubble(n.get("bubble_user_id")):
            logger.info("Check-in skipped: %s has a shift in Bubble in the last 30 days",
                        n.get("first_name"))
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


def _goodbye_message(first_name: str) -> str:
    name = (first_name or "").strip().split(" ")[0] or "there"
    return (f"Hi {name}, I didn't hear back so I've paused your account for now. We hate to "
            "see you go. If you ever want to change your mind, just reply YES. We now work "
            "across Wintringham, Brotherhood of St Laurence and VMCH so we have plenty of "
            "shifts to keep you busy =)")


def expire_no_replies() -> int:
    """Check-ins with no reply after NO_REPLY_HOURS count as a NO: the carer is marked
    inactive in Bubble and Supabase and sent a goodbye text, and Paul gets one summary
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
        # Belt and braces: any text from them since the check-in counts as a reply.
        replied = (client.table("helpdesk_messages").select("id")
                   .eq("phone", c["phone"]).eq("direction", "in")
                   .gte("created_at", c["sent_at"]).limit(1).execute().data or [])
        if replied:
            client.table("carer_checkins").update({"status": "replied"}).eq("id", c["id"]) \
                .eq("status", "sent").execute()
            continue
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
        bye = _goodbye_message(nurse.get("first_name"))
        try:
            db.send_sms(c["phone"], bye)
            _log(c["phone"], "out", bye, nurse_id=c["nurse_id"])
        except Exception:
            logger.exception("Goodbye text to %s failed", name)
    if names:
        msg = (f"Check-in: no reply after {NO_REPLY_HOURS}h, so I've marked "
               f"{len(names)} inactive: {', '.join(names)}.")
        if failed:
            msg += f" Couldn't update Bubble for {', '.join(failed)}, please do those by hand."
        _text_paul(msg)
    return len(names)


FOLLOWUP_DAYS = 2
FOLLOWUP_LAST_HOUR = 18  # follow-ups only go out 10am to 6pm Melbourne


def _followup_message(first_name: str) -> str:
    name = (first_name or "").strip().split(" ")[0] or "there"
    return (f"Hey {name}, just following up as I haven't seen your availability be updated? "
            "Were you still interested in working with us? If YES, that's great, just update "
            "your availability in the next couple days so we can send you shifts =)")


def send_followups() -> int:
    """Carers who replied YES to a check-in sent FOLLOWUP_DAYS+ ago: if they now have
    availability in Bubble, nothing is sent; if not, one follow-up text. Each check-in
    is followed up at most once (carer_checkins.followup_status)."""
    if db.DEV or db.MID:
        return 0
    now = datetime.now(MELB)
    if not (SEND_HOUR <= now.hour < FOLLOWUP_LAST_HOUR):
        return 0
    client = db.get_client()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=FOLLOWUP_DAYS)).isoformat()
    rows = (client.table("carer_checkins")
            .select("*, nurses(id, first_name, active, bubble_user_id)")
            .eq("status", "yes").is_("followup_status", "null")
            .lt("sent_at", cutoff).limit(50).execute().data or [])
    sent = 0
    for c in rows:
        nurse = c.get("nurses") or {}
        if not nurse.get("active"):
            continue
        has_avail = _has_future_availability_in_bubble(nurse.get("bubble_user_id"))
        if has_avail is None:
            continue  # Bubble lookup failed, try again next poll
        stamp = datetime.now(timezone.utc).isoformat()
        if has_avail:
            client.table("carer_checkins").update({
                "followup_status": "not_needed", "followup_at": stamp,
            }).eq("id", c["id"]).execute()
            continue
        claimed = (client.table("carer_checkins").update({
            "followup_status": "sent", "followup_at": stamp,
        }).eq("id", c["id"]).is_("followup_status", "null").execute().data or [])
        if not claimed:
            continue
        msg = _followup_message(nurse.get("first_name"))
        try:
            db.send_sms(c["phone"], msg)
            _log(c["phone"], "out", msg, nurse_id=c["nurse_id"])
            sent += 1
        except Exception:
            logger.exception("Follow-up text to %s failed", nurse.get("first_name"))
            client.table("carer_checkins").update({"followup_status": "failed"}) \
                .eq("id", c["id"]).execute()
        time.sleep(1)
    if sent:
        logger.info("Sent %d check-in follow-ups", sent)
    return sent


def final_checks() -> int:
    """FOLLOWUP_DAYS after the follow-up text: if the carer still has no availability in
    Bubble, set account status and active to no (Bubble and Supabase). No text to the
    carer; Paul gets one summary text."""
    if db.DEV or db.MID:
        return 0
    client = db.get_client()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=FOLLOWUP_DAYS)).isoformat()
    rows = (client.table("carer_checkins")
            .select("*, nurses(id, first_name, last_name, active, bubble_user_id)")
            .eq("followup_status", "sent").lt("followup_at", cutoff)
            .limit(50).execute().data or [])
    names, failed = [], []
    for c in rows:
        nurse = c.get("nurses") or {}
        has_avail = _has_future_availability_in_bubble(nurse.get("bubble_user_id"))
        if has_avail is None:
            continue  # Bubble lookup failed, try again next poll
        stamp = datetime.now(timezone.utc).isoformat()
        if has_avail or not nurse.get("active"):
            client.table("carer_checkins").update({
                "followup_status": "updated" if has_avail else "already_inactive",
                "followup_at": stamp,
            }).eq("id", c["id"]).execute()
            continue
        claimed = (client.table("carer_checkins").update({
            "followup_status": "deactivated", "followup_at": stamp,
        }).eq("id", c["id"]).eq("followup_status", "sent").execute().data or [])
        if not claimed:
            continue
        name = f"{nurse.get('first_name') or ''} {nurse.get('last_name') or ''}".strip() or "A carer"
        if not _deactivate_in_bubble(nurse.get("bubble_user_id")):
            failed.append(name)
        client.table("nurses").update({"active": False}).eq("id", c["nurse_id"]).execute()
        names.append(name)
    if names:
        msg = (f"Check-in: still no availability after the follow-up, so I've marked "
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
        try:
            send_followups()
        except Exception:
            logger.exception("Check-in follow-ups failed")
        try:
            final_checks()
        except Exception:
            logger.exception("Check-in final checks failed")
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
        # Anything other than a plain YES/NO ("Yes. Injured atm.", "available weekends")
        # still counts as a reply, so they are never paused for not replying. The
        # message itself carries on to the help desk, so Paul handles it.
        first = t.split(" ")[0] if t else ""
        db.get_client().table("carer_checkins").update({
            "status": "yes" if first in YES else "no" if first in NO else "replied",
            "reply": body, "replied_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", checkin["id"]).in_("status", ["sent"]).execute()
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
