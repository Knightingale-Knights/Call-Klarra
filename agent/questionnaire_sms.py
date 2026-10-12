"""
Participant questionnaire over SMS.

start()           worker calls this at shift start + 30 min: sends the first question.
handle_inbound()  sms_webhook calls this first for every inbound SMS. If the number has a
                  questionnaire In Progress, Claude reads the reply against the current
                  question, the answer is saved to Bubble + Supabase, and the next question
                  is returned as the SMS reply. Returns None when the number has none active.
check_timeouts()  worker calls this every minute: one reminder after 45 min of silence,
                  close after 90 min.

Rules:
  - One question per text; the next is only sent after the carer replies.
  - Any reply moves on. A reply that doesn't answer the question leaves it unanswered
    and flagged (in skipped_fields and in Paul's gaps text).
  - After 15 minutes of questions, the next reply gets a wrap-up text instead of another
    question. Remaining questions are picked up on later shifts.
  - When a session ends with gaps, Paul (KLARRA_DEV_PHONE) gets a text listing them.

Needs Supabase columns questionnaire_sessions.state (jsonb) and .last_message_at (timestamptz).
"""

import os
import re
import json
import logging
from datetime import datetime, timedelta, timezone

import anthropic

import db
import questionnaire as q

logger = logging.getLogger("knightingale-agent.questionnaire-sms")

MODEL = "claude-sonnet-4-6"
SESSION_LENGTH = timedelta(minutes=15)
REMIND_AFTER = timedelta(minutes=45)
CLOSE_AFTER = timedelta(minutes=90)
STOP_WORDS = {"stop", "finish", "busy", "not now"}


# ---------- helpers ----------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def effective_number(phone: str) -> str | None:
    """The number texts will actually reach, under KLARRA_MODE (same rules as db.send_sms)."""
    phone = db.normalise_phone(phone)
    if db.DEV and phone not in db.dev_testers():
        return db.normalise_phone(os.environ.get("KLARRA_DEV_PHONE"))
    if db.MID and phone not in db.dev_testers():
        return None
    return phone


def _claude():
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def _save_state(session: dict, state: dict, **fields) -> None:
    fields["state"] = state
    fields["last_message_at"] = _now().isoformat()
    db.get_client().table("questionnaire_sessions").update(fields).eq("id", session["id"]).execute()
    session.update(fields)


def _template_question(section: str, key: str, p: str) -> str:
    """Fallback wording if Claude is unavailable."""
    f = q.field_def(section, key)
    text = f"About {p}: {f['question']}"
    if f["kind"] == "list":
        text += f" Options include: {', '.join(q.OPTIONS[f['options']])}."
    elif f["kind"] == "answer":
        text += " (yes / no)"
    elif f["kind"] == "safety":
        text += " (safe / unsafe / not applicable)"
    return text


def compose_question(section: str, key: str, state: dict) -> str:
    """Natural SMS wording for one question, using the participant's name and pronouns."""
    p, pr = state["participant_name"], state["pronouns"]
    fallback = _template_question(section, key, p)
    try:
        msg = _claude().messages.create(
            model=MODEL, max_tokens=400,
            messages=[{"role": "user", "content": (
                f"Rewrite this as one short, friendly, plain-English text message to a care "
                f"worker who is with {p} right now. Refer to {p} by name or as "
                f"{pr['subj']}/{pr['obj']}/{pr['poss']}, never 'they'. Keep the full option list "
                f"if there is one. Don't greet, don't sign off, no emojis, never use an em dash. "
                f"Output only the message.\n\n{fallback}"
            )}],
        )
        text = msg.content[0].text.strip().replace("—", ", ")
        return text or fallback
    except Exception:
        logger.exception("Compose failed, using template")
        return fallback


def parse_reply(section: str, key: str, body: str, state: dict) -> dict:
    """Ask Claude whether the reply answers the question, and if so, the value to save."""
    f = q.field_def(section, key)
    if f["kind"] == "list":
        allowed = f"list; options: {', '.join(q.OPTIONS[f['options']])}"
    elif f["kind"] == "answer":
        allowed = "Yes / No / Unknown"
    elif f["kind"] == "safety":
        allowed = "Safe / Unsafe / N/A"
    else:
        allowed = "free text"
    prompt = f"""A care worker was texted this question about {state['participant_name']}, an NDIS participant:
"{f['question']}" [{allowed}]

Their reply: "{body}"

Return ONLY JSON: {{"answered": true, "value": "", "note": "", "stop": false}}

Rules:
- answered = false if the reply doesn't actually answer the question (e.g. "don't know", "skip", "ok", unrelated text). Never guess.
- free text: value is a short clear summary of what they said; "None" if nothing applies.
- Yes / No / Unknown and Safe / Unsafe / N/A: value is exactly one of those.
- list: value is comma separated option names where they fit, their own words for anything not in the options, or "None".
- note: if the answer is Unsafe, one short line saying what the problem is (e.g. "Stairs: loose railing"), else "".
- stop = true only if they say they can't continue now."""
    msg = _claude().messages.create(model=MODEL, max_tokens=400,
                                    messages=[{"role": "user", "content": prompt}])
    m = re.search(r"\{.*\}", msg.content[0].text, re.S)
    return json.loads(m.group(0)) if m else {"answered": False}


# ---------- session flow ----------

def start(session: dict, carer: dict, participant: dict, phone: str) -> None:
    """Send the first question. Marks the session In Progress."""
    section = session["section"]
    record = q.get_or_create_record(section, session["participant_bubble_id"])
    skipped = q.recently_skipped(session["participant_bubble_id"], section)
    fields = [k for k in q.missing_fields(section, record) if k not in skipped]
    if not fields:
        q.update_session(session, status="Completed", completed=_now().isoformat())
        return
    state = {
        "fields": fields, "pos": 0, "asked": [], "unanswered": [], "transcript": [],
        "record_id": record["_id"],
        "participant_name": q.first_name(participant), "carer_name": q.first_name(carer),
        "pronouns": q.pronouns(participant), "bubble_base": q.BUBBLE_BASE,
        "reminded": False, "started_at": _now().isoformat(),
    }
    p, c, pr = state["participant_name"], state["carer_name"], state["pronouns"]
    first = compose_question(section, fields[0], state)
    body = (f"Hi {c}, as promised here are a few quick questions about {p}. Please ask "
            f"{pr['obj']} and reply here. If you don't know an answer, just reply with anything "
            f"and I'll move on.\n\n{first}")
    state["asked"].append(fields[0])
    state["transcript"].append(f"Klarra: {body}")
    # Save first, then text: if saving fails nothing is sent (no repeat texts on retry).
    _save_state(session, state, carer_phone=phone, started=_now().isoformat())
    q.update_session(session, status="In Progress")
    db.send_sms(phone, body)
    logger.info("Questionnaire SMS started: %s / %s (%d questions)", p, section, len(fields))


def finish(session: dict, state: dict, closing: str | None = None) -> str | None:
    """Close the session, text Paul any gaps. Returns the closing text for the carer."""
    rec = q.get_record(session["section"], session["participant_bubble_id"]) or {}
    missing = set(q.missing_fields(session["section"], rec))
    still = [k for k in state["fields"] if k in missing]
    q.update_session(
        session,
        status="Completed" if not still else "Partial",
        completed=_now().isoformat(),
        skipped_fields=",".join(sorted(set(state["unanswered"]))),
        transcript="\n".join(state["transcript"])[:50000],
    )
    if still:
        text_gaps(session, state, still)
    return closing


def text_gaps(session: dict, state: dict, still: list[str]) -> None:
    admin = os.environ.get("KLARRA_DEV_PHONE")
    if not admin:
        return
    flagged = [k for k in still if k in set(state["unanswered"])]
    not_asked = [k for k in still if k not in set(state["asked"])]
    total = len(state["fields"])
    lines = [f"Questionnaire gaps for {state['participant_name']} ({session['section']}) "
             f"with {state['carer_name']}: {total - len(still)} of {total} answered."]
    if flagged:
        lines.append("Unanswered: " + ", ".join(flagged))
    if not_asked:
        lines.append("Not asked yet: " + ", ".join(not_asked))
    try:
        db.send_sms(admin, "\n".join(lines))
    except Exception:
        logger.exception("Failed to text questionnaire gaps")


def wrap_up_text(state: dict) -> str:
    return (f"Thanks for all those answers. I don't want to take up a ton of your time. I'll "
            f"leave you to support {state['participant_name']} now. Will touch base on your next "
            f"few shifts to get answers to the remaining Q's. Have a good one.")


def active_session(phone: str) -> dict | None:
    phone = db.normalise_phone(phone)
    if not phone:
        return None
    r = (db.get_client().table("questionnaire_sessions").select("*")
         .eq("status", "In Progress").eq("carer_phone", phone)
         .order("started", desc=True).limit(1).execute())
    return r.data[0] if r.data else None


def handle_inbound(phone: str, body: str) -> str | None:
    """Reply text for a carer mid-questionnaire, or None if this number has none active."""
    session = active_session(phone)
    if not session or not session.get("state"):
        return None
    state = session["state"]
    default_base = q.BUBBLE_BASE
    q.BUBBLE_BASE = state.get("bubble_base") or default_base
    try:
        return _handle(session, state, body)
    finally:
        q.BUBBLE_BASE = default_base


def _handle(session: dict, state: dict, body: str) -> str:
    section = session["section"]
    key = state["fields"][state["pos"]]
    c = state["carer_name"]
    state["transcript"].append(f"Carer: {body}")
    state["reminded"] = False

    if body.strip().lower().strip("!.? ") in STOP_WORDS:
        state["unanswered"].append(key)
        reply = f"No worries, thanks {c}. I'll pick up the rest on another shift."
        state["transcript"].append(f"Klarra: {reply}")
        return finish(session, state, reply)

    try:
        parsed = parse_reply(section, key, body, state)
    except Exception:
        logger.exception("Parse failed")
        parsed = {"answered": False}

    saved = False
    value = str(parsed.get("value") or "").strip()
    if parsed.get("answered") and value:
        try:
            fields, _ = q.normalise_answer(section, key, value)
            q.save_fields(section, state["record_id"], session["participant_bubble_id"], fields)
            saved = True
        except ValueError:
            pass
        note = (parsed.get("note") or "").strip()
        if saved and note and section == "Home Safety":
            q.save_fields(section, state["record_id"], session["participant_bubble_id"],
                          {"home safety notes": note}, append_notes=True)
    if not saved:
        state["unanswered"].append(key)  # flagged: reported to Paul, re-asked after 14 days

    if parsed.get("stop"):
        reply = f"No worries, thanks {c}. I'll pick up the rest on another shift."
        state["transcript"].append(f"Klarra: {reply}")
        return finish(session, state, reply)

    state["pos"] += 1
    if state["pos"] >= len(state["fields"]):
        reply = (f"That's everything, thanks so much {c}! Have a good rest of your shift "
                 f"with {state['participant_name']}.")
        state["transcript"].append(f"Klarra: {reply}")
        return finish(session, state, reply)

    if _now() - _ts(state["started_at"]) >= SESSION_LENGTH:
        reply = wrap_up_text(state)
        state["transcript"].append(f"Klarra: {reply}")
        return finish(session, state, reply)

    nxt_key = state["fields"][state["pos"]]
    reply = f"Thanks! {compose_question(section, nxt_key, state)}"
    state["asked"].append(nxt_key)
    state["transcript"].append(f"Klarra: {reply}")
    _save_state(session, state)
    return reply


def check_timeouts() -> None:
    """Remind once after 45 min of silence, close after 90 min."""
    r = (db.get_client().table("questionnaire_sessions").select("*")
         .eq("status", "In Progress").execute())
    now = _now()
    for session in r.data or []:
        state = session.get("state") or {}
        last = session.get("last_message_at")
        if not state or not last:
            continue
        idle = now - _ts(last)
        default_base = q.BUBBLE_BASE
        q.BUBBLE_BASE = state.get("bubble_base") or default_base
        try:
            if idle >= CLOSE_AFTER:
                key = state["fields"][state["pos"]]
                if key not in state["unanswered"]:
                    state["unanswered"].append(key)
                state["transcript"].append("(closed after no reply)")
                finish(session, state)
            elif idle >= REMIND_AFTER and not state.get("reminded"):
                msg = (f"Hi {state['carer_name']}, just checking in. Are you able to answer the "
                       f"question above about {state['participant_name']}?")
                db.send_sms(session["carer_phone"], msg)
                state["reminded"] = True
                state["transcript"].append(f"Klarra: {msg}")
                db.get_client().table("questionnaire_sessions").update(
                    {"state": state}).eq("id", session["id"]).execute()
        except Exception:
            logger.exception("Timeout handling failed for session %s", session["id"])
        finally:
            q.BUBBLE_BASE = default_base
