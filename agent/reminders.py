"""
Personal reminders for Paul, set by texting Klarra in plain language.

  "I have a call with Jane Doe Wed 21 at 9am, text me 20 mins before"

admin_agent.py's set_reminder tool calls add(); a background thread started by
sms_orchestrator.py (start_worker) texts each reminder when it falls due.

Only reachable from Paul's number: admin_agent only runs for the admin phone, so a
carer texting "remind me..." never touches this.

Table: public.reminders (id, phone, message, remind_at, status, source_text,
sent_at, created_at, followups, last_nudged_at, done_at).
status: pending -> sent -> done | failed | cancelled.
Once sent, Klarra follows up every 30 min (up to 3 times) until Paul texts "done"
(mark_done, called by sms_webhook ahead of the admin agent).
"""

import time
import logging
import threading
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import db

logger = logging.getLogger("knightingale-reminders")

MELB = ZoneInfo("Australia/Melbourne")
POLL_SECONDS = 20


def _fmt(dt_utc: datetime) -> str:
    """'8:40am Wed 21 Oct' in Melbourne time."""
    local = dt_utc.astimezone(MELB)
    hour = local.strftime("%I").lstrip("0")
    return f"{hour}:{local.strftime('%M')}{local.strftime('%p').lower()} {local.strftime('%a %d %b').replace(' 0', ' ')}"


def _parse_utc(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def add(phone: str, message: str, remind_at_local: str, source_text: str | None = None) -> dict:
    """remind_at_local: 'YYYY-MM-DD HH:MM' in Melbourne time."""
    try:
        local = datetime.strptime(remind_at_local.strip(), "%Y-%m-%d %H:%M").replace(tzinfo=MELB)
    except ValueError:
        return {"error": "remind_at must be YYYY-MM-DD HH:MM (Melbourne time)"}
    remind_utc = local.astimezone(timezone.utc)
    if remind_utc <= datetime.now(timezone.utc):
        return {"error": f"That time ({_fmt(remind_utc)}) has already passed."}
    message = (message or "").strip()
    if not message:
        return {"error": "Reminder text is empty."}

    row = db.get_client().table("reminders").insert({
        "phone": db.normalise_phone(phone),
        "message": message,
        "remind_at": remind_utc.isoformat(),
        "source_text": source_text,
    }).execute().data[0]
    return {
        "ok": True,
        "id": row["id"],
        "reply_text": f"Done. I'll text you at {_fmt(remind_utc)}: {message}",
    }


def list_pending(phone: str) -> dict:
    rows = (db.get_client().table("reminders").select("id,message,remind_at")
            .eq("phone", db.normalise_phone(phone)).eq("status", "pending")
            .order("remind_at").limit(20).execute().data) or []
    if not rows:
        return {"reminders": [], "reply_text": "No reminders set."}
    lines = [f"{i}. {_fmt(_parse_utc(r['remind_at']))}: {r['message']}" for i, r in enumerate(rows, 1)]
    return {
        "reminders": [{"number": i, **r} for i, r in enumerate(rows, 1)],
        "reply_text": "Upcoming reminders:\n" + "\n".join(lines),
    }


def cancel(phone: str, reminder_id: str) -> dict:
    rows = (db.get_client().table("reminders").update({"status": "cancelled"})
            .eq("id", reminder_id).eq("phone", db.normalise_phone(phone))
            .eq("status", "pending").execute().data) or []
    if not rows:
        return {"error": "No pending reminder with that id."}
    r = rows[0]
    return {"ok": True, "reply_text": f"Cancelled: {r['message']} ({_fmt(_parse_utc(r['remind_at']))})"}


def send_due() -> int:
    """Text every pending reminder whose time has come. Returns how many were sent."""
    client = db.get_client()
    now = datetime.now(timezone.utc).isoformat()
    due = (client.table("reminders").select("id")
           .eq("status", "pending").lte("remind_at", now)
           .order("remind_at").limit(50).execute().data) or []
    sent = 0
    for r in due:
        # Claim first (pending -> sent) so a reminder can never go out twice.
        claimed = (client.table("reminders")
                   .update({"status": "sent", "sent_at": datetime.now(timezone.utc).isoformat()})
                   .eq("id", r["id"]).eq("status", "pending").execute().data) or []
        if not claimed:
            continue
        row = claimed[0]
        try:
            db.send_sms(row["phone"], f"Reminder: {row['message']}\nText me \"done\" when it's sorted.")
            sent += 1
        except Exception:
            logger.exception("Reminder %s failed to send", row["id"])
            client.table("reminders").update({"status": "failed"}).eq("id", row["id"]).execute()
    return sent


FOLLOWUP_MINUTES = 30
MAX_FOLLOWUPS = 3
DONE_WORDS = {"done", "all done", "its done", "it's done", "sorted", "done thanks",
              "done thank you", "completed", "finished", "did it", "done it"}


def send_followups() -> int:
    """A reminder Paul hasn't replied 'done' to gets a follow-up every FOLLOWUP_MINUTES,
    up to MAX_FOLLOWUPS times."""
    client = db.get_client()
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=FOLLOWUP_MINUTES)).isoformat()
    oldest = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()
    rows = (client.table("reminders").select("*")
            .eq("status", "sent").lt("followups", MAX_FOLLOWUPS)
            .lte("sent_at", cutoff).gte("sent_at", oldest).limit(50).execute().data) or []
    nudged = 0
    for r in rows:
        last = r.get("last_nudged_at") or r.get("sent_at")
        if last and _parse_utc(last) > datetime.now(timezone.utc) - timedelta(minutes=FOLLOWUP_MINUTES):
            continue
        n = (r.get("followups") or 0) + 1
        claimed = (client.table("reminders")
                   .update({"followups": n, "last_nudged_at": datetime.now(timezone.utc).isoformat()})
                   .eq("id", r["id"]).eq("followups", n - 1).eq("status", "sent")
                   .execute().data) or []
        if not claimed:
            continue
        last_one = n >= MAX_FOLLOWUPS
        body = f"Following up: {r['message']}. Text me \"done\" when it's sorted."
        if last_one:
            body = f"Last nudge on this one: {r['message']}. Text me \"done\" when it's sorted."
        try:
            db.send_sms(r["phone"], body)
            nudged += 1
        except Exception:
            logger.exception("Reminder follow-up %s failed", r["id"])
    return nudged


def mark_done(phone: str, body: str) -> str | None:
    """Paul texting 'done': close his most recent reminder still waiting on it.
    Returns the reply, or None if the text isn't 'done' or nothing is waiting."""
    t = " ".join((body or "").lower().replace("!", " ").replace(".", " ").split())
    if t not in DONE_WORDS:
        return None
    client = db.get_client()
    rows = (client.table("reminders").select("*")
            .eq("phone", db.normalise_phone(phone)).eq("status", "sent")
            .gte("sent_at", (datetime.now(timezone.utc) - timedelta(days=1)).isoformat())
            .order("sent_at", desc=True).limit(1).execute().data) or []
    if not rows:
        return None
    r = rows[0]
    client.table("reminders").update({
        "status": "done", "done_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", r["id"]).execute()
    return f"Nice one. Marked done: {r['message']}"


def _loop():
    logger.info("Reminder worker running, polling every %ss.", POLL_SECONDS)
    while True:
        try:
            send_due()
        except Exception:
            logger.exception("Reminder poll failed")
        try:
            send_followups()
        except Exception:
            logger.exception("Reminder follow-up poll failed")
        time.sleep(POLL_SECONDS)


def start_worker() -> None:
    """Run the reminder sender on its own thread so long shift cascades never delay it."""
    threading.Thread(target=_loop, name="reminders", daemon=True).start()
