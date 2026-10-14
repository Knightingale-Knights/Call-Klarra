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


# Fields that are skipped (saved as the given value) when the field they depend on is "None".
DEPENDS = {
    "last hospital stay length": ("hospital admissions", "N/A"),
    "emergency contact phone two": ("emergency contact name two", "None"),
    "emergency contact relationship two": ("emergency contact name two", "None"),
}


def _template_question(section: str, key: str, p: str, gate: bool = False) -> str:
    """Fallback wording if Claude is unavailable."""
    f = q.field_def(section, key)
    text = f"About {p}: {f['gate'] if gate else f['question']}"
    if gate or f["kind"] == "answer":
        text += " (yes / no)"
    elif f["kind"] == "safety":
        text += " (safe / unsafe / not applicable)"
    return text


def _bullets(items: list[str]) -> str:
    return "\n".join(f"\u2022 {'No, none' if i == 'None' else i}" for i in items)


# Fixed wording for list questions. {p} name, {subj_is} "she's"/"he's"/"they're".
LIST_TEMPLATES = {
    "physical health": "Does {p} have any of the physical health conditions listed below? Please reply with the names of the conditions:",
    "mental health": "Does {p} have any of the mental health conditions listed below? Please reply with the names of the conditions:",
    "personal care": "Does {p} need help with any of the personal care tasks listed below? Please reply with the names of the tasks:",
    "nutrition": "Do you believe any of the eating or drinking concerns listed below apply to {p}? Please reply with the names of the concerns:",
    "accidental movement": "Does {p} have any of the accidental movements listed below? Please reply with the names of the movements:",
    "manual handling": "Does {p} need any of the manual handling assistance listed below? Please reply with the names of the types of assistance:",
    "environmental risks": "Do you believe any of the hazards listed below pose a threat to {p}'s safety when {subj_is} at home? Please reply with the names of the hazards:",
    "financial risks": "Do you believe any of the financial risks listed below apply to {p}? Please reply with the names of the risks:",
    "social risks": "Do you believe any of the social risks listed below pose a threat to {p}'s safety? Please reply with the names of the risks:",
    "substance use": "Does {p} use any of the substances listed below? Please reply with the names of the substances:",
}


MAX_OPTIONS_PER_TEXT = 12


def list_parts(f: dict) -> list[list[str]]:
    """Long option lists are split into texts of up to 12 options (plus "No, none")."""
    opts = [o for o in q.OPTIONS[f["options"]] if o != "None"]
    n = max(1, -(-len(opts) // MAX_OPTIONS_PER_TEXT))
    size = -(-len(opts) // n)
    return [opts[i:i + size] for i in range(0, len(opts), size)]


def _list_question(key: str, f: dict, state: dict) -> str:
    pr = state["pronouns"]
    subj_is = {"he": "he's", "she": "she's"}.get(pr["subj"], "they're")
    tpl = LIST_TEMPLATES.get(key) or f"{f['question']} Please reply with the names:"
    parts = list_parts(f)
    part = min(state.get("part", 0), len(parts) - 1)
    return (tpl.format(p=state["participant_name"], subj_is=subj_is) + "\n"
            + _bullets(["None"] + parts[part]))


def compose_question(section: str, key: str, state: dict, gate: bool = False) -> str:
    """Natural SMS wording for one question, using the participant's name and pronouns.
    List questions get their options as dot points underneath."""
    p, pr = state["participant_name"], state["pronouns"]
    f = q.field_def(section, key)
    if not gate and f["kind"] == "list":
        return _list_question(key, f, state)
    question = _template_question(section, key, p, gate)
    try:
        msg = _claude().messages.create(
            model=MODEL, max_tokens=300,
            messages=[{"role": "user", "content": (
                f"Rewrite this as one short, friendly, plain-English question in a text message "
                f"to a care worker who is with {p} right now. Refer to {p} by name or as "
                f"{pr['subj']}/{pr['obj']}/{pr['poss']}, never 'they'. Ask exactly what is asked "
                f"and nothing more: don't add details, examples or extra requests, and don't turn "
                f"a 'list' question into a yes/no one. Don't list any options, don't greet, don't "
                f"sign off, no emojis, never use an em dash. Output only the question.\n\n{question}"
            )}],
        )
        question = msg.content[0].text.strip().replace("\u2014", ", ") or question
    except Exception:
        logger.exception("Compose failed, using template")
    return question


def ask(section: str, state: dict) -> str:
    """Wording for the current field: its yes/no gate first if it has one.
    New questions get a "Question N:" header; a gate's follow-up doesn't."""
    key = state["fields"][state["pos"]]
    follow_up = state.get("phase") == "main" and key in state["asked"]
    gate = bool(q.field_def(section, key).get("gate")) and state.get("phase") != "main"
    state["phase"] = "gate" if gate else "main"
    if key not in state["asked"]:
        state["asked"].append(key)
    text = compose_question(section, key, state, gate=gate)
    f = q.field_def(section, key)
    n = state["asked"].index(key) + 1
    if f["kind"] == "list" and len(list_parts(f)) > 1:
        total = len(list_parts(f))
        return f"Question {n} (part {state.get('part', 0) + 1} of {total}):\n{text}"
    if follow_up:
        return text
    return f"Question {n}:\n{text}"


def parse_gate(body: str, gate_q: str, p: str) -> dict:
    """Yes / no / unknown for a gate question, plus any details given with a yes."""
    prompt = f"""A care worker was texted this yes/no question about {p}: "{gate_q}"
Their reply: "{body}"
Return ONLY JSON: {{"answer": "yes", "detail": "", "stop": false}}
- answer: "yes", "no", or "unknown" (unknown if they don't know or the reply doesn't answer it).
- detail: if they said yes AND already gave the details (e.g. named the items), a short clear summary of them; else "".
- stop = true only if they say they can't continue now."""
    msg = _claude().messages.create(model=MODEL, max_tokens=300,
                                    messages=[{"role": "user", "content": prompt}])
    m = re.search(r"\{.*\}", msg.content[0].text, re.S)
    return json.loads(m.group(0)) if m else {"answer": "unknown"}


def parse_reply(section: str, key: str, body: str, state: dict,
                options: list[str] | None = None) -> dict:
    """Ask Claude whether the reply answers the question, and if so, the value to save."""
    f = q.field_def(section, key)
    if f["kind"] == "list":
        allowed = f"list; options: {', '.join(options or q.OPTIONS[f['options']])}"
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
        "phase": None, "none_fields": [],
    }
    p, c, pr = state["participant_name"], state["carer_name"], state["pronouns"]
    first = ask(section, state)
    body = (f"Hi {c}, as promised here are a few quick questions about {p}. Please complete "
            f"the questionnaire together. If you don't know an answer, just say \"I don't "
            f"know\" and I'll move on.\n\n{first}")
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
    # The worker emails Paul the gaps (one email per session), see email_pending_gaps().
    state["gaps"] = still
    state["gaps_emailed"] = False if still else True
    db.get_client().table("questionnaire_sessions").update(
        {"state": state}).eq("id", session["id"]).execute()
    return closing


def gaps_email(session: dict, state: dict) -> tuple[str, str]:
    """(subject, text body) listing every question left unanswered in this session."""
    section = session["section"]
    still = state.get("gaps") or []
    flagged = set(state.get("unanswered") or [])
    asked = set(state.get("asked") or [])
    p, c = state["participant_name"], state["carer_name"]
    total = len(state["fields"])

    def line(k):
        f = q.field_def(section, k)
        return f"- {f['question'] if f else k}  [{k}]"

    dont_know = [k for k in still if k in flagged]
    no_reply = [k for k in still if k not in flagged and k in asked]
    not_asked = [k for k in still if k not in asked]
    out = [f"Participant: {p}", f"Section: {section}", f"Carer: {c}",
           f"Answered: {total - len(still)} of {total}", ""]
    if dont_know:
        out += ["Carer didn't know / didn't answer:"] + [line(k) for k in dont_know] + [""]
    if no_reply:
        out += ["Carer didn't reply:"] + [line(k) for k in no_reply] + [""]
    if not_asked:
        out += ["Not asked yet (session ended):"] + [line(k) for k in not_asked] + [""]
    out.append("These will be asked again on a later shift unless you fill them in Bubble first.")
    return f"Questionnaire follow-up: {p} ({section})", "\n".join(out)


def send_email(subject: str, body: str) -> None:
    """Postmark. Needs POSTMARK_SERVER_TOKEN and QUESTIONNAIRE_EMAIL_FROM (a verified sender)."""
    import requests
    r = requests.post(
        "https://api.postmarkapp.com/email",
        headers={"X-Postmark-Server-Token": os.environ["POSTMARK_SERVER_TOKEN"],
                 "Accept": "application/json", "Content-Type": "application/json"},
        json={"From": os.environ["QUESTIONNAIRE_EMAIL_FROM"],
              "To": os.environ.get("QUESTIONNAIRE_EMAIL_TO", "paul@knightingale.com.au"),
              "Subject": subject, "TextBody": body, "MessageStream": "outbound"},
        timeout=20,
    )
    r.raise_for_status()


def email_pending_gaps() -> None:
    """Worker: one email per finished session that ended with unanswered questions."""
    since = (_now() - timedelta(days=2)).isoformat()
    r = (db.get_client().table("questionnaire_sessions").select("*")
         .in_("status", ["Completed", "Partial"]).gte("completed", since).execute())
    for session in r.data or []:
        state = session.get("state") or {}
        if not state or state.get("gaps_emailed") is not False:
            continue
        try:
            subject, body = gaps_email(session, state)
            send_email(subject, body)
            state["gaps_emailed"] = True
            db.get_client().table("questionnaire_sessions").update(
                {"state": state}).eq("id", session["id"]).execute()
            logger.info("Emailed gaps for session %s", session["id"])
        except Exception:
            logger.exception("Gaps email failed for session %s", session["id"])


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


def _save(session: dict, state: dict, key: str, value: str, note: str = "") -> bool:
    """Normalise and save one answer. Returns False if the value isn't valid for the field."""
    section = session["section"]
    try:
        fields, _ = q.normalise_answer(section, key, value)
    except ValueError:
        return False
    try:
        q.save_fields(section, state["record_id"], session["participant_bubble_id"], fields)
    except Exception:
        # Never drop the carer out of the questionnaire over a save error: flag and move on.
        logger.exception("Save failed for %s / %s", section, key)
        return False
    if str(fields.get(key)).strip().lower() in ("none", "['none']"):
        state.setdefault("none_fields", []).append(key)
    if note and section == "Home Safety":
        q.save_fields(section, state["record_id"], session["participant_bubble_id"],
                      {"home safety notes": note}, append_notes=True)
    return True


def _advance(session: dict, state: dict) -> None:
    """Move to the next field, auto-filling ones whose parent answer was None."""
    state["pos"] += 1
    state["phase"] = None
    while state["pos"] < len(state["fields"]):
        key = state["fields"][state["pos"]]
        dep = DEPENDS.get(key)
        if dep and dep[0] in state.get("none_fields", []):
            _save(session, state, key, dep[1])
            state["pos"] += 1
            continue
        break


def _handle(session: dict, state: dict, body: str) -> str:
    section = session["section"]
    key = state["fields"][state["pos"]]
    f = q.field_def(section, key)
    c, p = state["carer_name"], state["participant_name"]
    state["transcript"].append(f"Carer: {body}")
    state["reminded"] = False
    stop_reply = f"No worries, thanks {c}. I'll pick up the rest on another shift."

    if body.strip().lower().strip("!.? ") in STOP_WORDS:
        state["unanswered"].append(key)
        state["transcript"].append(f"Klarra: {stop_reply}")
        return finish(session, state, stop_reply)

    stop = False
    if state.get("phase") == "gate":
        try:
            g = parse_gate(body, f["gate"], p)
        except Exception:
            logger.exception("Gate parse failed")
            g = {"answer": "unknown"}
        stop = bool(g.get("stop"))
        answer = str(g.get("answer") or "").lower()
        detail = str(g.get("detail") or "").strip()
        if answer == "no":
            _save(session, state, key, "None")
        elif answer == "yes" and detail:
            if not _save(session, state, key, detail):
                state["unanswered"].append(key)
        elif answer == "yes" and not stop:
            # Yes, but no details yet: ask the follow-up for the same field.
            state["phase"] = "main"
            reply = f"Thanks! {ask(section, state)}"  # gate follow-up, same question
            state["transcript"].append(f"Klarra: {reply}")
            _save_state(session, state)
            return reply
        else:
            state["unanswered"].append(key)
    elif f["kind"] == "list" and len(list_parts(f)) > 1:
        parts = list_parts(f)
        part = state.get("part", 0)
        try:
            parsed = parse_reply(section, key, body, state, ["None"] + parts[part])
        except Exception:
            logger.exception("Parse failed")
            parsed = {"answered": False}
        stop = bool(parsed.get("stop"))
        value = str(parsed.get("value") or "").strip()
        if parsed.get("answered") and value:
            picked = [v.strip() for v in value.split(",")
                      if v.strip() and v.strip().lower() not in ("none", "no", "no, none")]
            state.setdefault("part_values", []).extend(picked)
        else:
            state["part_unknown"] = True
        if part + 1 < len(parts) and not stop:
            # Same question, next part of the list.
            state["part"] = part + 1
            reply = f"Thanks!\n\n{ask(section, state)}"
            state["transcript"].append(f"Klarra: {reply}")
            _save_state(session, state)
            return reply
        values = state.pop("part_values", [])
        unknown = state.pop("part_unknown", False)
        state["part"] = 0
        saved = _save(session, state, key, ", ".join(values) if values else "None") \
            if (values or not unknown) else False
        if unknown or not saved:
            state["unanswered"].append(key)  # partly or not answered: flagged to Paul
    else:
        try:
            parsed = parse_reply(section, key, body, state)
        except Exception:
            logger.exception("Parse failed")
            parsed = {"answered": False}
        stop = bool(parsed.get("stop"))
        value = str(parsed.get("value") or "").strip()
        ok = bool(parsed.get("answered") and value) and _save(
            session, state, key, value, (parsed.get("note") or "").strip())
        if not ok:
            state["unanswered"].append(key)  # flagged: reported to Paul, re-asked after 14 days

    if stop:
        state["transcript"].append(f"Klarra: {stop_reply}")
        return finish(session, state, stop_reply)

    _advance(session, state)
    if state["pos"] >= len(state["fields"]):
        reply = f"That's everything, thanks so much {c}! Have a good rest of your shift with {p}."
        state["transcript"].append(f"Klarra: {reply}")
        return finish(session, state, reply)

    if _now() - _ts(state["started_at"]) >= SESSION_LENGTH:
        reply = wrap_up_text(state)
        state["transcript"].append(f"Klarra: {reply}")
        return finish(session, state, reply)

    reply = f"Thanks!\n\n{ask(section, state)}"
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
