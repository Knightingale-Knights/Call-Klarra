"""
Participant questionnaire over SMS.

start()           worker calls this at shift start + 30 min: sends the first question.
handle_inbound()  sms_webhook calls this first for every inbound SMS. If the number has a
                  questionnaire In Progress, Claude reads the reply, answers are saved to
                  Bubble + Supabase, and the next question is returned as the SMS reply.
                  Returns None when the number has no active questionnaire.
check_timeouts()  worker calls this every minute: one reminder after 45 min of silence,
                  close as Partial after 90 min.

Questions go one per text, except Yes/No and Safe/Unsafe questions, which go in groups of 5.
When a session ends with gaps, Paul (KLARRA_DEV_PHONE) gets a text listing them.

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
GROUP_SIZE = 5
REMIND_AFTER = timedelta(minutes=45)
CLOSE_AFTER = timedelta(minutes=90)
SKIP_WORDS = {"skip", "dont know", "don't know", "idk", "unsure", "not sure", "no idea", "dunno"}
STOP_WORDS = {"stop", "finish", "end", "busy", "later", "not now"}


# ---------- helpers ----------

def _now() -> datetime:
    return datetime.now(timezone.utc)


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


def build_queue(section: str, fields: list[str]) -> list[list[str]]:
    """One text per group: consecutive Yes/No or Safe/Unsafe questions in groups of 5,
    consecutive free-text questions on the same topic together (e.g. emergency contacts),
    list questions alone."""
    queue, group, group_key = [], [], None
    for k in fields:
        f = q.field_def(section, k)
        if f["kind"] == "list":
            key, limit = None, 1
        elif f["kind"] in ("answer", "safety"):
            key, limit = f["kind"], GROUP_SIZE
        else:
            key, limit = ("text", f["topic"]), 6
        if group and (key is None or key != group_key or len(group) >= limit):
            queue.append(group)
            group = []
        group.append(k)
        group_key = key
    if group:
        queue.append(group)
    return queue


def _template_question(section: str, keys: list[str], p: str) -> str:
    """Fallback wording if Claude is unavailable."""
    defs = [q.field_def(section, k) for k in keys]
    if len(defs) == 1:
        f = defs[0]
        text = f"About {p}: {f['question']}"
        if f["kind"] == "list":
            text += f" Options include: {', '.join(q.OPTIONS[f['options']])}."
        elif f["kind"] == "answer":
            text += " (yes / no / don't know)"
        elif f["kind"] == "safety":
            text += " (safe / unsafe / not applicable)"
        return text
    items = "\n".join(f"{i}. {f['question']}" for i, f in enumerate(defs, 1))
    if defs[0]["kind"] in ("text", "notes"):
        return f"About {p}:\n{items}"
    if defs[0]["kind"] == "safety":
        return (f"A few quick checks at {p}'s place. Reply 'all fine', or tell me which numbers "
                f"aren't safe and why:\n{items}")
    return (f"A few quick yes/no ones about {p}'s home. Reply 'none', or the numbers that "
            f"apply:\n{items}")


def compose_question(section: str, keys: list[str], state: dict) -> str:
    """Natural SMS wording for the next question(s), using the participant's name and pronouns."""
    p, pr = state["participant_name"], state["pronouns"]
    fallback = _template_question(section, keys, p)
    try:
        msg = _claude().messages.create(
            model=MODEL, max_tokens=500,
            messages=[{"role": "user", "content": (
                f"Rewrite this as one friendly, plain-English text message from Klarra (an "
                f"assistant at Knightingale, a care agency) to a carer who is with {p} right now. "
                f"Refer to {p} by name or as {pr['subj']}/{pr['obj']}/{pr['poss']}, never 'they'. "
                f"Keep any numbering and the full option list if there is one. Don't greet, don't "
                f"sign off, no emojis, never use an em dash. Output only the message.\n\n{fallback}"
            )}],
        )
        text = msg.content[0].text.strip()
        return text.replace("—", ", ") or fallback
    except Exception:
        logger.exception("Compose failed, using template")
        return fallback


def parse_reply(section: str, keys: list[str], body: str, state: dict) -> dict:
    """Ask Claude to turn the carer's reply into field values for the current question(s)."""
    lines = []
    for k in keys:
        f = q.field_def(section, k)
        if f["kind"] == "list":
            allowed = f"list; options: {', '.join(q.OPTIONS[f['options']])}"
        elif f["kind"] == "answer":
            allowed = "Yes / No / Unknown"
        elif f["kind"] == "safety":
            allowed = "Safe / Unsafe / N/A"
        else:
            allowed = "free text"
        lines.append(f'- "{k}": {f["question"]} [{allowed}]')
    prompt = f"""You record a care worker's text-message answers to questions about {state['participant_name']}, an NDIS participant.

Questions just asked:
{chr(10).join(lines)}

Carer's reply: "{body}"

Return ONLY JSON:
{{"answers": {{"<field>": "<value>"}}, "dont_know": ["<field>"], "notes": "", "stop": false, "clarify": ""}}

Rules:
- free text: a short clear summary of what they said; "None" if nothing applies.
- Yes / No / Unknown: exactly one of those. For a group, "none"/"no" means all No; listed numbers or items are Yes and the rest No.
- Safe / Unsafe / N/A: exactly one of those. For a group, "all fine"/"all good" means all Safe; listed numbers are Unsafe unless they say not applicable.
- list: comma separated option names where they fit, the carer's own words for anything not in the options, or "None".
- notes: for anything Unsafe, one short line per item saying what the problem is (e.g. "Stairs: loose railing"), else "".
- If they don't know a field, put it in dont_know. Never guess or invent answers.
- stop = true only if they say they can't continue now.
- If the reply doesn't answer the question(s) or is unclear, leave answers empty and put a short polite follow-up question in clarify. Never use an em dash."""
    msg = _claude().messages.create(model=MODEL, max_tokens=800,
                                    messages=[{"role": "user", "content": prompt}])
    text = msg.content[0].text
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {"clarify": "Sorry, could you say that another way?"}


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
        "fields": fields, "queue": build_queue(section, fields), "pos": 0,
        "skipped": [], "transcript": [], "record_id": record["_id"],
        "participant_name": q.first_name(participant), "carer_name": q.first_name(carer),
        "pronouns": q.pronouns(participant), "bubble_base": q.BUBBLE_BASE,
        "reminded": False, "clarify_count": 0,
    }
    p, c, pr = state["participant_name"], state["carer_name"], state["pronouns"]
    first = compose_question(section, state["queue"][0], state)
    total = len(state["queue"])
    body = (f"Hi {c}, it's Klarra from Knightingale. As promised, here are a few quick questions "
            f"about {p}. Please ask {pr['obj']} and reply here. Reply 'skip' if you don't know "
            f"one, or 'stop' to finish for today.\n\n(1/{total}) {first}")
    state["transcript"].append(f"Klarra: {body}")
    # Save first, then text: if saving fails nothing is sent (no repeat texts on retry).
    _save_state(session, state, carer_phone=phone, started=_now().isoformat())
    q.update_session(session, status="In Progress")
    db.send_sms(phone, body)
    logger.info("Questionnaire SMS started: %s / %s (%d questions)", p, section, total)


def finish(session: dict, state: dict, closing: str | None = None) -> str | None:
    """Close the session, text Paul any gaps. Returns the closing text for the carer."""
    rec = q.get_record(session["section"], session["participant_bubble_id"]) or {}
    missing = set(q.missing_fields(session["section"], rec))
    still = [k for k in state["fields"] if k in missing]
    q.update_session(
        session,
        status="Completed" if not still else "Partial",
        completed=_now().isoformat(),
        skipped_fields=",".join(sorted(set(state["skipped"]))),
        transcript="\n".join(state["transcript"])[:50000],
    )
    if still:
        text_gaps(session, state, still)
    return closing


def text_gaps(session: dict, state: dict, still: list[str]) -> None:
    admin = os.environ.get("KLARRA_DEV_PHONE")
    if not admin:
        return
    skipped = set(state["skipped"])
    unknown = [k for k in still if k in skipped]
    not_answered = [k for k in still if k not in skipped]
    total = len(state["fields"])
    lines = [f"Questionnaire gaps for {state['participant_name']} ({session['section']}) "
             f"with {state['carer_name']}: {total - len(still)} of {total} answered."]
    if unknown:
        lines.append("Didn't know: " + ", ".join(unknown))
    if not_answered:
        lines.append("Not answered: " + ", ".join(not_answered))
    try:
        db.send_sms(admin, "\n".join(lines))
    except Exception:
        logger.exception("Failed to text questionnaire gaps")


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
    keys = state["queue"][state["pos"]]
    c, p = state["carer_name"], state["participant_name"]
    state["transcript"].append(f"Carer: {body}")
    state["reminded"] = False
    words = body.strip().lower().strip("!.? ")

    if words in STOP_WORDS:
        state["skipped"] = state["skipped"]  # unanswered stay as "not answered"
        reply = f"No worries, thanks {c}. We'll pick up the rest on another shift."
        state["transcript"].append(f"Klarra: {reply}")
        return finish(session, state, reply)

    if words in SKIP_WORDS:
        parsed = {"answers": {}, "dont_know": list(keys)}
    else:
        try:
            parsed = parse_reply(section, keys, body, state)
        except Exception:
            logger.exception("Parse failed")
            parsed = {"clarify": "Sorry, I didn't catch that. Could you reply again?"}

    if parsed.get("stop"):
        reply = f"No worries, thanks {c}. We'll pick up the rest on another shift."
        state["transcript"].append(f"Klarra: {reply}")
        return finish(session, state, reply)

    answers = {k: v for k, v in (parsed.get("answers") or {}).items() if k in keys and v}
    dont_know = [k for k in (parsed.get("dont_know") or []) if k in keys]
    clarify = (parsed.get("clarify") or "").strip()

    if not answers and not dont_know and clarify:
        state["clarify_count"] = state.get("clarify_count", 0) + 1
        if state["clarify_count"] < 2:
            state["transcript"].append(f"Klarra: {clarify}")
            _save_state(session, state)
            return clarify
        dont_know = list(keys)  # second unclear reply: skip and move on

    # Save answers (one Bubble write for the group).
    to_write = {}
    for k, v in answers.items():
        try:
            fields, _ = q.normalise_answer(section, k, str(v))
            to_write.update(fields)
        except ValueError:
            dont_know.append(k)
    if to_write:
        q.save_fields(section, state["record_id"], session["participant_bubble_id"], to_write)
    notes = (parsed.get("notes") or "").strip()
    if notes and section == "Home Safety":
        q.save_fields(section, state["record_id"], session["participant_bubble_id"],
                      {"home safety notes": notes}, append_notes=True)
    for k in keys:
        if k not in answers and k not in dont_know:
            dont_know.append(k)  # anything in the group left unanswered counts as unknown
    state["skipped"] = sorted(set(state["skipped"]) | set(dont_know))

    # Next question or finish.
    state["pos"] += 1
    state["clarify_count"] = 0
    if state["pos"] >= len(state["queue"]):
        reply = (f"That's everything for today, thanks so much {c}! Anything we missed will be "
                 f"picked up on another shift.")
        state["transcript"].append(f"Klarra: {reply}")
        return finish(session, state, reply)
    nxt = compose_question(section, state["queue"][state["pos"]], state)
    reply = f"Thanks! ({state['pos'] + 1}/{len(state['queue'])}) {nxt}"
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
        idle = now - datetime.fromisoformat(last.replace("Z", "+00:00"))
        default_base = q.BUBBLE_BASE
        q.BUBBLE_BASE = state.get("bubble_base") or default_base
        try:
            if idle >= CLOSE_AFTER:
                state["transcript"].append("(closed after no reply)")
                finish(session, state)
            elif idle >= REMIND_AFTER and not state.get("reminded"):
                msg = (f"Hi {state['carer_name']}, just checking in. Are you able to answer the "
                       f"question above about {state['participant_name']}? Reply 'stop' if now "
                       f"isn't a good time.")
                db.send_sms(session["carer_phone"], msg)
                state["reminded"] = True
                state["transcript"].append(f"Klarra: {msg}")
                db.get_client().table("questionnaire_sessions").update(
                    {"state": state}).eq("id", session["id"]).execute()
        except Exception:
            logger.exception("Timeout handling failed for session %s", session["id"])
        finally:
            q.BUBBLE_BASE = default_base
