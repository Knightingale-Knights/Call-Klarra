"""
Carer help desk over SMS. Klarra sits in the middle of every conversation between Paul
and carers, and checks every answer with Paul before a carer sees it.

Carer -> Klarra (any text from a known carer not caught by another flow):
  1. The text is embedded and the 3 closest saved answers are pulled from helpdesk_kb
     (match_helpdesk_kb RPC). The recent conversation with this carer (helpdesk_messages)
     is included, so replies to Paul's relayed messages make sense. She can also look
     things up in every Supabase table (read-only, klarra_data.query_database), e.g.
     the carer's own roster, before drafting.
  2. Claude (fast model) decides:
       draft   - she has an answer (from saved answers / the conversation). It is NOT
                 sent. The carer gets "Just a sec, working on this." and Paul gets
                 "Q12 Kelly said: ... I'd reply: ...".
       unknown - she has no answer. Carer gets the holding text, Paul gets
                 "Q12 from Kelly: ...".
       ack     - a bare acknowledgement ("ok", "will do"). Nothing goes back to the
                 carer; Paul gets an FYI if it's a reply to something he sent.

Paul -> Klarra:
  "Q12 yes"            send the draft as is (and count it as an approval)
  "Q12 no"             she asks him for the right answer
  "Q12 <answer>"       she learns it (new saved answer, or replaces the one she used if
                       it was wrong), rewrites it as an SMS and texts the carer
  "tell Kelly ..."     admin_agent's message_carer tool -> relay_to_carer(): she texts
                       the carer in her own voice, logs it, and saves anything general
                       as a saved answer
  "? <question>"       test the help desk as if he were a carer

Tables: helpdesk_kb (approvals / corrections counted per saved answer),
helpdesk_questions (one row per carer text, status pending | draft | answered |
approved | corrected | ack, plus a topic from TOPICS), helpdesk_messages (full carer
conversation log). View helpdesk_topics: mode, approve vs correct rate and current
approval streak per topic. Topics hand over to Klarra after 20 approvals in a row
(see "topic handover" below); helpdesk_topic_settings holds each topic's mode.
If embeddings fail (e.g. OpenAI quota) it falls back to the most recent saved answers.
"""

import os
import re
import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import anthropic

import db
import klarra_data

logger = logging.getLogger("knightingale-helpdesk")

MODEL = os.environ.get("HELPDESK_MODEL", "claude-haiku-4-5").strip()
FALLBACK_MODEL = "claude-sonnet-4-6"
ADMIN_PHONE = "+61426512584"
HOLDING_REPLY = "Just a sec, working on this."
TOP_K = 3
FALLBACK_KB_ROWS = 40
HISTORY_DAYS = 7
HISTORY_MESSAGES = 8
MELB = ZoneInfo("Australia/Melbourne")

# Topics used to measure how often Paul approves vs corrects Klarra
# (view helpdesk_topic_stats). Keep this list stable: the stats group by it.
TOPICS = [
    "shifts_roster",          # when/where am I working, shift details, start times
    "shift_changes",          # swaps, cancellations, picking up or dropping shifts
    "sick_late",              # sick, running late, can't make it
    "pay_timesheets",         # pay, rates, timesheets, invoices, super, kms
    "checkin_app",            # check-in/out, the app, links, progress note submission
    "participants_care",      # how to support a participant, care questions
    "incidents_safety",       # incidents, injuries, safety concerns, complaints
    "policies_procedures",    # policies, procedures, rules, uniform, conduct
    "training_compliance",    # training, certificates, checks, documents
    "other",
]

TOPIC_LABELS = {
    "shifts_roster": "shifts and roster", "shift_changes": "shift changes",
    "sick_late": "sick or late", "pay_timesheets": "pay and timesheets",
    "checkin_app": "check-in and the app", "participants_care": "participant care",
    "incidents_safety": "incidents and safety", "policies_procedures": "policies",
    "training_compliance": "training", "other": "other",
}
MODES = ("review", "auto", "never_auto")

ANSWER_PATTERN = re.compile(r"^\s*q\s*(\d+)\b[\s:.\-]*(.*)$", re.IGNORECASE | re.DOTALL)
YES_WORDS = {"yes", "y", "yep", "yeah", "yup", "good", "great", "correct", "ok", "okay",
             "perfect", "spot on", "send", "send it", "good answer", "all good", "approved",
             "thats right", "that's right", "right", "\U0001F44D"}
NO_WORDS = {"no", "n", "nope", "nah", "wrong", "incorrect", "don't send", "dont send"}


# --- helpers ---------------------------------------------------------------

def _claude(system: str, user: str, max_tokens: int = 400) -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    last_err = None
    for model in (MODEL, FALLBACK_MODEL):
        try:
            msg = client.messages.create(
                model=model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in msg.content if b.type == "text").strip()
        except Exception as e:
            last_err = e
            logger.warning("Help desk model %s failed: %s", model, e)
    raise last_err


TRIAGE_MAX_STEPS = 6


def _claude_with_data(system: str, user: str, max_tokens: int = 600) -> str:
    """Like _claude, but Claude may look things up with query_database first.
    Returns the final text."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    last_err = None
    for model in (MODEL, FALLBACK_MODEL):
        try:
            messages = [{"role": "user", "content": user}]
            for _ in range(TRIAGE_MAX_STEPS):
                resp = client.messages.create(
                    model=model, max_tokens=max_tokens, system=system,
                    tools=[klarra_data.TOOL], messages=messages,
                )
                if resp.stop_reason != "tool_use":
                    return "".join(b.text for b in resp.content if b.type == "text").strip()
                messages.append({"role": "assistant", "content": resp.content})
                results = []
                for b in resp.content:
                    if b.type == "tool_use":
                        logger.info("Help desk lookup: %s", (b.input or {}).get("sql"))
                        results.append({"type": "tool_result", "tool_use_id": b.id,
                                        "content": klarra_data.call(b.input)})
                messages.append({"role": "user", "content": results})
            # Out of steps: ask for the answer without tools.
            messages.append({"role": "user", "content": "Give your final JSON answer now."})
            resp = client.messages.create(model=model, max_tokens=max_tokens,
                                          system=system, messages=messages)
            return "".join(b.text for b in resp.content if b.type == "text").strip()
        except Exception as e:
            last_err = e
            logger.warning("Help desk model %s failed: %s", model, e)
    raise last_err


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}


def _clean(text: str) -> str:
    return (text or "").replace("—", ",").replace("–", "-").strip()


def _words(text: str) -> str:
    return (text or "").lower().strip(" !.,")


def _embed(text: str) -> list[float] | None:
    try:
        return db.embed_situation(text)
    except Exception:
        logger.exception("Help desk embedding failed")
        return None


def _backfill_embeddings() -> None:
    """Embed saved answers that don't have an embedding yet."""
    client = db.get_client()
    rows = (client.table("helpdesk_kb").select("id, question, answer")
            .is_("embedding", "null").eq("active", True).limit(50).execute().data or [])
    for r in rows:
        emb = _embed(f"{r['question']}\n{r['answer']}")
        if emb is None:
            return
        client.table("helpdesk_kb").update({"embedding": emb}).eq("id", r["id"]).execute()


def _candidates(text: str) -> list[dict]:
    client = db.get_client()
    try:
        _backfill_embeddings()
    except Exception:
        logger.exception("Help desk embedding backfill failed")
    emb = _embed(text)
    if emb is not None:
        try:
            r = client.rpc("match_helpdesk_kb", {"p_embedding": emb, "p_limit": TOP_K}).execute()
            return r.data or []
        except Exception:
            logger.exception("match_helpdesk_kb failed, using fallback KB list")
    r = (client.table("helpdesk_kb").select("id, question, answer").eq("active", True)
         .order("created_at", desc=True).limit(FALLBACK_KB_ROWS).execute())
    return r.data or []


def _save_kb(question: str, answer: str, source: str, question_id: int | None = None,
             replace_id: int | None = None, topic: str | None = None) -> int | None:
    client = db.get_client()
    payload = {"question": question.strip(), "answer": _clean(answer),
               "embedding": _embed(f"{question}\n{answer}")}
    if topic:
        payload["topic"] = topic
    try:
        if replace_id:
            client.table("helpdesk_kb").update(payload).eq("id", replace_id).execute()
            return replace_id
        row = client.table("helpdesk_kb").insert({
            **payload, "source": source, "source_question_id": question_id,
        }).execute().data[0]
        return row["id"]
    except Exception:
        logger.exception("Failed to save help desk answer")
        return None


def _bump_kb(kb_id: int | None, field: str) -> None:
    if not kb_id:
        return
    try:
        client = db.get_client()
        r = client.table("helpdesk_kb").select(field).eq("id", kb_id).limit(1).execute()
        if r.data:
            client.table("helpdesk_kb").update(
                {field: (r.data[0].get(field) or 0) + 1}).eq("id", kb_id).execute()
    except Exception:
        logger.exception("Failed to update help desk %s", field)


def carer_for(phone: str) -> dict | None:
    """The nurses row for this number, or None if it isn't a known carer."""
    phone = db.normalise_phone(phone)
    if not phone:
        return None
    r = (db.get_client().table("nurses").select("id, first_name, last_name, phone")
         .eq("phone", phone).limit(1).execute())
    return r.data[0] if r.data else None


def _log(phone: str, direction: str, body: str, nurse_id=None, source=None,
         question_id=None) -> None:
    try:
        db.get_client().table("helpdesk_messages").insert({
            "phone": db.normalise_phone(phone), "nurse_id": nurse_id,
            "direction": direction, "body": body, "source": source,
            "question_id": question_id,
        }).execute()
    except Exception:
        logger.exception("Failed to log help desk message")


def _history(phone: str) -> str:
    """Recent conversation with this carer, oldest first, as plain lines."""
    try:
        since = (datetime.now(timezone.utc) - timedelta(days=HISTORY_DAYS)).isoformat()
        rows = (db.get_client().table("helpdesk_messages")
                .select("direction, body, created_at")
                .eq("phone", db.normalise_phone(phone)).gte("created_at", since)
                .order("created_at", desc=True).limit(HISTORY_MESSAGES).execute().data or [])
    except Exception:
        logger.exception("Failed to load help desk history")
        return ""
    lines = []
    for r in reversed(rows):
        who = "Carer" if r["direction"] == "in" else "Klarra"
        lines.append(f"{who}: {r['body']}")
    return "\n".join(lines)


def _last_out_was_relay(phone: str) -> bool:
    """True if the last thing we texted this carer was a real message (a relay from
    Paul or an answer he approved), not just the holding text."""
    try:
        r = (db.get_client().table("helpdesk_messages").select("source")
             .eq("phone", db.normalise_phone(phone)).eq("direction", "out")
             .order("created_at", desc=True).limit(1).execute().data or [])
        return bool(r) and r[0].get("source") in ("relay", "approved", "paul")
    except Exception:
        return False


def _text_paul(body: str) -> None:
    try:
        db.send_sms(ADMIN_PHONE, body)
    except Exception:
        logger.exception("Failed to text Paul from help desk")


def _send_carer(q: dict, text: str, source: str) -> bool:
    try:
        db.send_sms(q["phone"], text)
    except Exception:
        logger.exception("Failed to text carer for Q%s", q.get("id"))
        return False
    _log(q["phone"], "out", text, nurse_id=q.get("nurse_id"), source=source,
         question_id=q.get("id"))
    return True


# --- topic handover ----------------------------------------------------------
#
# Each topic is in one of three modes (helpdesk_topic_settings):
#   review     - every draft goes to Paul first (the default)
#   auto       - Klarra sends confident, non-sensitive drafts herself and FYIs Paul
#   never_auto - always goes to Paul, never switches to auto (incidents_safety)
# A review topic switches to auto by itself once Paul has approved `threshold` (20)
# drafts in a row. Any correction of an auto-sent answer switches it back to review.

def topic_mode(topic: str) -> str:
    try:
        r = (db.get_client().table("helpdesk_topic_settings").select("mode")
             .eq("topic", topic).limit(1).execute().data or [])
        return r[0]["mode"] if r else "review"
    except Exception:
        logger.exception("Could not read topic mode")
        return "review"


def set_topic_mode(topic: str, mode: str, by: str = "paul") -> dict:
    topic = (topic or "").strip().lower()
    if topic not in TOPICS:
        # Accept the friendly label too ("sick or late").
        by_label = {v: k for k, v in TOPIC_LABELS.items()}
        topic = by_label.get(topic, topic)
    if topic not in TOPICS:
        return {"error": f"unknown topic. Topics: {', '.join(TOPIC_LABELS.values())}"}
    if mode not in MODES:
        return {"error": f"mode must be one of {', '.join(MODES)}"}
    db.get_client().table("helpdesk_topic_settings").upsert({
        "topic": topic, "mode": mode, "changed_at": _now(), "changed_by": by,
    }).execute()
    return {"topic": topic, "label": TOPIC_LABELS[topic], "mode": mode}


def _topic_stats(topic: str) -> dict:
    try:
        r = (db.get_client().table("helpdesk_topics").select("*")
             .eq("topic", topic).limit(1).execute().data or [])
        return r[0] if r else {}
    except Exception:
        logger.exception("Could not read topic stats")
        return {}


def _maybe_hand_over(topic: str | None) -> str:
    """After an approval: switch the topic to auto once the streak reaches the
    threshold. Returns a note for Paul, or ''."""
    if not topic:
        return ""
    st = _topic_stats(topic)
    if st.get("mode") != "review":
        return ""
    streak, threshold = st.get("approved_streak") or 0, st.get("threshold") or 20
    label = TOPIC_LABELS.get(topic, topic)
    if streak >= threshold:
        set_topic_mode(topic, "auto", by="streak")
        return (f"\n\nThat's {streak} {label} answers in a row you've approved, so I'll "
                f"send those myself from now on and FYI you. Text \"check {label} with me\" "
                "to switch back.")
    if threshold - streak <= 3:
        return f"\n({label}: {streak}/{threshold} in a row before I take these over.)"
    return ""


# --- carer side ------------------------------------------------------------

TRIAGE_SYSTEM = (
    "You are Klarra, the SMS assistant for Knightingale, a Melbourne aged care and NDIS "
    "nursing staffing agency run by Paul. A carer (nurse or support worker) has texted. "
    "You get the recent conversation with them and a few saved answers Paul has taught "
    "you. Draft the reply you would send. Decide kind:\n"
    "  draft = the saved answers, the database or the conversation give you what you need to reply "
    "(questions, and also situations like being sick, running late, can't make a shift, "
    "pay or timesheet issues). Write the reply using ONLY what the saved answers, the "
    "database and the conversation say. Never guess or add facts.\n"
    "  unknown = you don't have what you need to reply. If in doubt, choose unknown.\n"
    "  ack = a bare acknowledgement with nothing to answer (ok, will do, thanks, noted).\n"
    "You can look things up in the database with query_database (read-only): their "
    "own shifts, roster, hours, availability and offers, the participants and sites "
    "they work with, policies. Look up what you need rather than guessing. You are told "
    "the carer's nurses.id. Only share information about this carer, their own shifts, "
    "and what they need to know for the people and sites they work with. Never share "
    "other carers' personal details (phone, address, pay, availability) or participants' "
    "health details beyond what the carer needs for their shift.\n"
    "used_kb_id: the [number] of the saved answer you relied on, or null.\n"
    "topic: one of " + ", ".join(TOPICS) + ".\n"
    "confident: true only if every fact in your reply comes straight from a saved "
    "answer, the database or the conversation, and it fully answers them.\n"
    "sensitive: true if it involves an injury, incident, abuse or neglect, a participant's "
    "health, medication or wellbeing, a complaint, a safety risk, or anything legal or "
    "disciplinary. These always go to Paul.\n"
    "Replies are SMS: warm, plain, brief, Australian English, no markdown, no em dashes, "
    "use their first name at most once. Never say or imply you checked, asked or "
    "confirmed with Paul; just give the answer.\n"
    'Respond ONLY with JSON: {"kind":"draft|unknown|ack","used_kb_id":number|null,'
    '"topic":"...","confident":true|false,"sensitive":true|false,"reply":"text or empty"}'
)


def handle_carer_text(phone: str, body: str, carer: dict | None = None,
                      test: bool = False) -> str | None:
    """Handle a carer's text. Returns the SMS reply for the carer ("" for no reply),
    or None if this isn't a known carer (caller carries on)."""
    body = (body or "").strip()
    if not body:
        return None
    carer = carer or carer_for(phone)
    if not carer:
        return None
    name = carer.get("first_name") or "Carer"
    who = f"{name} {carer.get('last_name') or ''}".strip() + (" (test)" if test else "")

    history = _history(phone)
    was_relay = _last_out_was_relay(phone)
    _log(phone, "in", body, nurse_id=carer.get("id"), source="test" if test else None)

    candidates = _candidates(body)
    kb_text = "\n\n".join(
        f"[{c['id']}] Q: {c['question']}\nA: {c['answer']}" for c in candidates
    ) or "(no saved answers yet)"

    try:
        now = datetime.now(timezone.utc).astimezone(MELB)
        decision = _json(_claude_with_data(
            TRIAGE_SYSTEM + klarra_data.schema_prompt(),
            f"Today is {now.strftime('%A %d %B %Y, %H:%M')} Melbourne time.\n"
            f"Carer: {who} (nurses.id {carer.get('id')}, phone {db.normalise_phone(phone)})\n\n"
            f"Recent conversation:\n{history or '(none)'}\n\n"
            f"Saved answers:\n{kb_text}\n\nCarer's new text: \"{body}\"",
            max_tokens=600,
        ))
    except Exception:
        logger.exception("Help desk triage failed, escalating")
        decision = {}

    kind = (decision.get("kind") or "unknown").lower()
    draft = _clean(decision.get("reply"))
    used = decision.get("used_kb_id")
    used = used if isinstance(used, int) and any(c.get("id") == used for c in candidates) else None

    topic = (decision.get("topic") or "other").strip().lower()
    if topic not in TOPICS:
        topic = "other"

    client = db.get_client()
    row = {
        "phone": db.normalise_phone(phone),
        "nurse_id": carer.get("id"),
        "carer_name": name,
        "question": body,
        "kind": "general",
        "topic": topic,
    }

    if kind == "ack":
        _record_ack(row, who, was_relay or test)
        return ""

    can_auto = (
        kind == "draft" and draft
        and decision.get("confident") is True and decision.get("sensitive") is False
        and topic_mode(topic) == "auto"
        and not db.MID
    )
    if can_auto:
        q = client.table("helpdesk_questions").insert({
            **row, "status": "auto_sent", "draft_reply": draft, "reply_sent": draft,
            "kb_id": used, "answered_at": _now(),
        }).execute().data[0]
        _log(phone, "out", HOLDING_REPLY, nurse_id=carer.get("id"), source="holding",
             question_id=q["id"])
        if _send_carer(q, draft, "auto"):
            _text_paul(
                f"Q{q['id']} FYI I answered {who} ({TOPIC_LABELS.get(topic, topic)}):\n"
                f"They said: {body}\nI replied: {draft}\n\n"
                f"Reply Q{q['id']} + the right answer if that's wrong."
            )
            return HOLDING_REPLY
        # Text failed: fall back to asking Paul (holding text already logged above).
        client.table("helpdesk_questions").update({"status": "draft", "reply_sent": None,
                                                   "answered_at": None}).eq("id", q["id"]).execute()
        _text_paul(
            f"Q{q['id']} {who} said:\n{body}\n\nI'd reply:\n{draft}\n\n"
            f"Reply Q{q['id']} yes, or Q{q['id']} + the right answer."
        )
    elif kind == "draft" and draft:
        q = client.table("helpdesk_questions").insert({
            **row, "status": "draft", "draft_reply": draft, "kb_id": used,
        }).execute().data[0]
        _text_paul(
            f"Q{q['id']} {who} said:\n{body}\n\nI'd reply:\n{draft}\n\n"
            f"Reply Q{q['id']} yes, or Q{q['id']} + the right answer."
        )
    else:
        q = client.table("helpdesk_questions").insert({**row, "status": "pending"}).execute().data[0]
        _text_paul(f"Q{q['id']} from {who}:\n{body}\n\nReply Q{q['id']} + your answer.")

    if not can_auto:
        _log(phone, "out", HOLDING_REPLY, nurse_id=carer.get("id"), source="holding",
             question_id=q["id"])
    return HOLDING_REPLY


ACK_RE = re.compile(
    r"^(ok(ay)?|k|kk|thanks?( you)?( so much)?|thankyou|ty|ta|cheers|will do|noted|got it|"
    r"sounds good|no worries|no problem|all good|perfect|great|cool|done|yep|yes|yeah|no|"
    r"nope|sure|of course|\U0001F44D|\U0001F64F|\U0001F60A)[\s!.]*$",
    re.IGNORECASE,
)


def _record_ack(row: dict, who: str, tell_paul: bool) -> None:
    db.get_client().table("helpdesk_questions").insert({
        **row, "kind": "chat", "status": "ack", "answered_at": _now(),
    }).execute()
    if tell_paul:
        _text_paul(f"{who} replied: {row['question']}")


def start_carer_text(phone: str, body: str, carer: dict | None = None,
                     test: bool = False) -> str | None:
    """Webhook entry point. Twilio only waits ~15s, and drafting can take longer when
    Klarra looks things up, so: a bare acknowledgement is handled now (no reply), and
    anything else gets the holding text straight away while the drafting and Paul's
    text happen in the background. Returns None if this isn't a known carer."""
    body = (body or "").strip()
    if not body:
        return None
    carer = carer or carer_for(phone)
    if not carer:
        return None
    if ACK_RE.match(body):
        name = carer.get("first_name") or "Carer"
        who = f"{name} {carer.get('last_name') or ''}".strip() + (" (test)" if test else "")
        tell = _last_out_was_relay(phone) or test
        _log(phone, "in", body, nurse_id=carer.get("id"), source="test" if test else None)
        _record_ack({"phone": db.normalise_phone(phone), "nurse_id": carer.get("id"),
                     "carer_name": name, "question": body}, who, tell)
        return ""

    def work():
        try:
            handle_carer_text(phone, body, carer=carer, test=test)
        except Exception:
            logger.exception("Help desk background handling failed")
            _text_paul(f"Help desk error on a text from {carer.get('first_name')}: {body}")

    threading.Thread(target=work, daemon=True).start()
    return HOLDING_REPLY


# --- Paul answering --------------------------------------------------------

COMPOSE_SYSTEM = (
    "You are Klarra, the SMS assistant for Knightingale (a Melbourne aged care and NDIS "
    "staffing agency). A carer texted, and Paul (the director) has told you how to "
    "respond. His answer may be terse. You may also see a reply you drafted and the "
    "saved answer it came from. The carer has NOT seen your draft, so never refer to it "
    "or to correcting anything. Write:\n"
    "reply = the SMS you, Klarra, send the carer. Use ONLY what Paul said; never add "
    "facts, conditions or advice. Warm, plain, brief, Australian English, no markdown, "
    "no em dashes. Never say or imply you checked, asked or confirmed with Paul; just "
    "give the answer.\n"
    "kb_question = the carer's message as a general situation or question any carer "
    "could send (no names, dates or details specific to this carer), e.g. \"What should "
    "I do if I'm sick and can't make my shift?\".\n"
    "kb_answer = how to respond to kb_question in future, general and standalone, using "
    "only what Paul said (drop anything only true for this carer this time).\n"
    "saved_answer_wrong = true only if a saved answer was shown and Paul's answer shows "
    "it is wrong (so it should be replaced); false otherwise.\n"
    'Respond ONLY with JSON: {"reply":"...","kb_question":"...","kb_answer":"...",'
    '"saved_answer_wrong":true|false}'
)


def handle_paul_answer(body: str) -> str | None:
    """Paul's 'Q12 ...' text. Returns his confirmation, or None if the text isn't a help
    desk answer (caller carries on)."""
    m = ANSWER_PATTERN.match(body or "")
    if not m:
        return None
    qid, answer = int(m.group(1)), m.group(2).strip()

    client = db.get_client()
    r = client.table("helpdesk_questions").select("*").eq("id", qid).limit(1).execute()
    if not r.data:
        return None  # not a help desk question number: let normal routing have it
    q = r.data[0]
    name = q.get("carer_name") or "the carer"

    if q["status"] == "auto_sent":
        if not answer or _words(answer) in YES_WORDS:
            client.table("helpdesk_questions").update(
                {"paul_answer": answer or "ok"}).eq("id", qid).execute()
            return f"Thanks, noted Q{qid} as right."
        if db.MID:
            return "KLARRA_MODE is mid, so carers can't be texted. Nothing sent."
        return _correct_auto_answer(q, answer)
    if q["status"] not in ("pending", "draft"):
        return f"Q{qid} is already sorted."
    if not answer:
        return f"Reply Q{qid} yes, or Q{qid} + the right answer."
    if db.MID:
        return "KLARRA_MODE is mid, so carers can't be texted. Nothing sent."

    w = _words(answer)

    # Approve the draft as is.
    if q["status"] == "draft" and w in YES_WORDS:
        draft = q.get("draft_reply") or ""
        if not _send_carer(q, draft, "approved"):
            return f"Couldn't text {name} just now. Try Q{qid} yes again in a minute."
        kb_id = q.get("kb_id")
        if kb_id:
            _bump_kb(kb_id, "approvals")
        else:
            # Drafted from the conversation, not a saved answer: learn it now.
            out = _json(_safe_compose(q, draft, None))
            if out.get("kb_question") and out.get("kb_answer"):
                kb_id = _save_kb(out["kb_question"], out["kb_answer"], "approved_draft", qid,
                                 topic=q.get("topic"))
                _bump_kb(kb_id, "approvals")
        client.table("helpdesk_questions").update({
            "status": "approved", "paul_answer": answer, "reply_sent": draft,
            "kb_id": kb_id, "answered_at": _now(),
        }).eq("id", qid).execute()
        return f"Sent to {name}." + _maybe_hand_over(q.get("topic"))

    if w in NO_WORDS:
        return f"What should I tell {name}? Reply Q{qid} + the right answer."
    if q["status"] == "pending" and w in YES_WORDS:
        return f"I don't have an answer for Q{qid} yet. Reply Q{qid} + what to tell {name}."

    # Paul's own answer (for a question she couldn't answer, or a correction of a draft).
    saved = None
    if q.get("kb_id"):
        s = (client.table("helpdesk_kb").select("id, question, answer")
             .eq("id", q["kb_id"]).limit(1).execute().data or [])
        saved = s[0] if s else None

    out = _json(_safe_compose(q, answer, saved))
    reply = _clean(out.get("reply") or answer)

    kb_q = out.get("kb_question") or q["question"]
    kb_a = out.get("kb_answer") or answer
    replace_id = saved["id"] if saved and out.get("saved_answer_wrong") else None
    kb_id = _save_kb(kb_q, kb_a, "paul", qid, replace_id=replace_id, topic=q.get("topic"))
    if q["status"] == "draft" and saved:
        _bump_kb(saved["id"], "corrections")

    if not _send_carer(q, reply, "paul"):
        client.table("helpdesk_questions").update({"kb_id": kb_id}).eq("id", qid).execute()
        return f"Learnt it, but couldn't text {name} just now. Try Q{qid} again in a minute."

    client.table("helpdesk_questions").update({
        "status": "corrected" if q["status"] == "draft" else "answered",
        "paul_answer": answer, "reply_sent": reply, "kb_id": kb_id, "answered_at": _now(),
    }).eq("id", qid).execute()

    learnt = "Updated what I knew" if replace_id else "Learnt it"
    if not kb_id:
        learnt = "Couldn't save it"
    return f"{learnt} and texted {name}:\n{reply}"


def _safe_compose(q: dict, answer: str, saved: dict | None) -> str:
    try:
        return _claude(
            COMPOSE_SYSTEM,
            f"Carer: {q.get('carer_name')}\n\nRecent conversation:\n"
            f"{_history(q['phone']) or '(none)'}\n\n"
            f"Carer's message: \"{q['question']}\"\n"
            f"Your draft: \"{q.get('draft_reply') or 'none'}\"\n"
            f"Saved answer used: {json.dumps(saved) if saved else 'none'}\n\n"
            f"Paul's answer: \"{answer}\"",
            max_tokens=500,
        )
    except Exception:
        logger.exception("Help desk compose failed, using Paul's answer as written")
        return ""


# --- Paul messaging a carer through Klarra ---------------------------------

RELAY_SYSTEM = (
    "You are Klarra, the SMS assistant for Knightingale (a Melbourne aged care and NDIS "
    "staffing agency). Paul (the director) has asked you to pass a message to a carer. "
    "Write:\n"
    "text = the SMS you, Klarra, send the carer. Keep everything Paul said, add nothing. "
    "Warm, plain, brief, Australian English, no markdown, no em dashes. Use their first "
    "name once. You may say Paul asked you to pass it on.\n"
    "kb_question / kb_answer = if the message contains general information other carers "
    "might ask about (a policy, procedure, uniform, pay rule, how something works), a "
    "general question and standalone answer capturing it, using only what Paul said. "
    "Otherwise null for both (for example personal or one-off messages about this "
    "carer's own shift).\n"
    'Respond ONLY with JSON: {"text":"...","kb_question":"... or null","kb_answer":"... or null"}'
)


def relay_to_carer(nurse: dict, message: str) -> dict:
    """Text a carer a message from Paul, in Klarra's voice. Used by admin_agent."""
    message = (message or "").strip()
    if not message:
        return {"error": "No message to send."}
    if db.MID:
        return {"error": "KLARRA_MODE is mid, so carers cannot be contacted. Nothing sent."}
    if not nurse.get("phone"):
        return {"error": f"No phone number for {nurse.get('first_name')}."}

    name = nurse.get("first_name") or "there"
    try:
        out = _json(_claude(
            RELAY_SYSTEM,
            f"Carer first name: {name}\n\nRecent conversation:\n"
            f"{_history(nurse['phone']) or '(none)'}\n\nPaul's message: \"{message}\"",
            max_tokens=400,
        ))
    except Exception:
        logger.exception("Relay compose failed, sending Paul's words")
        out = {}
    text = _clean(out.get("text") or message)

    try:
        db.send_sms(nurse["phone"], text)
    except Exception as e:
        logger.exception("Relay SMS failed")
        return {"error": f"Text to {name} failed: {e}"}
    _log(nurse["phone"], "out", text, nurse_id=nurse.get("id"), source="relay")

    learnt = False
    kq, ka = out.get("kb_question"), out.get("kb_answer")
    if kq and ka and str(kq).lower() != "null" and str(ka).lower() != "null":
        learnt = _save_kb(kq, ka, "relay") is not None

    return {"sent_to": f"{name} {nurse.get('last_name') or ''}".strip(), "text": text,
            "learnt_general_info": learnt}


# --- Paul correcting an answer Klarra sent on her own ---------------------------

CORRECTION_SYSTEM = (
    "You are Klarra, the SMS assistant for Knightingale (a Melbourne aged care and NDIS "
    "staffing agency). You already texted a carer an answer on your own, and Paul (the "
    "director) says it was wrong. His correction may be terse. Write:\n"
    "reply = a follow-up SMS to the carer, starting \"Quick update on my last message:\", "
    "giving the right answer. Use ONLY what Paul said. Warm, plain, brief, Australian "
    "English, no markdown, no em dashes. Never say or imply you checked, asked or "
    "confirmed with Paul.\n"
    "kb_question = the carer's message as a general question any carer could send.\n"
    "kb_answer = the right general answer, using only what Paul said.\n"
    "saved_answer_wrong = true if the saved answer shown is itself wrong (replace it), "
    "false if it was fine but didn't fit this message.\n"
    'Respond ONLY with JSON: {"reply":"...","kb_question":"...","kb_answer":"...",'
    '"saved_answer_wrong":true|false}'
)


def _correct_auto_answer(q: dict, answer: str) -> str:
    client = db.get_client()
    qid, name = q["id"], q.get("carer_name") or "the carer"
    saved = None
    if q.get("kb_id"):
        r = (client.table("helpdesk_kb").select("id, question, answer")
             .eq("id", q["kb_id"]).limit(1).execute().data or [])
        saved = r[0] if r else None
    try:
        out = _json(_claude(
            CORRECTION_SYSTEM,
            f"Carer: {name}\nCarer's message: \"{q['question']}\"\n"
            f"What you sent: \"{q.get('reply_sent')}\"\n"
            f"Saved answer used: {json.dumps(saved) if saved else 'none'}\n\n"
            f"Paul's correction: \"{answer}\"",
            max_tokens=500,
        ))
    except Exception:
        logger.exception("Correction compose failed, using Paul's words")
        out = {}

    replace_id = saved["id"] if saved and out.get("saved_answer_wrong") else None
    kb_id = _save_kb(out.get("kb_question") or q["question"], out.get("kb_answer") or answer,
                     "paul", qid, replace_id=replace_id, topic=q.get("topic"))
    if saved:
        _bump_kb(saved["id"], "corrections")

    reply = _clean(out.get("reply") or f"Quick update on my last message: {answer}")
    sent = _send_carer(q, reply, "paul")
    client.table("helpdesk_questions").update({
        "status": "auto_corrected", "paul_answer": answer, "kb_id": kb_id,
        "reply_sent": reply if sent else q.get("reply_sent"),
    }).eq("id", qid).execute()

    topic = q.get("topic") or "other"
    note = ""
    if topic_mode(topic) == "auto":
        set_topic_mode(topic, "review", by="correction")
        note = f"\nI'll check {TOPIC_LABELS.get(topic, topic)} answers with you again for now."
    learnt = "Updated what I knew" if replace_id else "Learnt it"
    if not sent:
        return f"{learnt}, but couldn't text {name} the correction." + note
    return f"{learnt} and texted {name}:\n{reply}" + note


# --- timeout: don't leave carers hanging ------------------------------------------

TIMEOUT_MINUTES = int(os.environ.get("HELPDESK_TIMEOUT_MINUTES", "15"))
TIMEOUT_REPLY = "Checking with Paul, back to you soon."
TIMEOUT_POLL_SECONDS = 60


def check_timeouts() -> int:
    """Any carer text still waiting on Paul after TIMEOUT_MINUTES: text the carer once
    that Klarra is checking with Paul, and nudge Paul once. Returns how many."""
    if db.MID:
        return 0
    client = db.get_client()
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=TIMEOUT_MINUTES)).isoformat()
    rows = (client.table("helpdesk_questions").select("*")
            .in_("status", ["pending", "draft"]).is_("timeout_sent_at", "null")
            .lt("created_at", cutoff).limit(20).execute().data or [])
    for q in rows:
        # Claim it first so two workers never double-text.
        claimed = (client.table("helpdesk_questions").update({"timeout_sent_at": _now()})
                   .eq("id", q["id"]).is_("timeout_sent_at", "null").execute().data or [])
        if not claimed:
            continue
        if q["phone"] != db.normalise_phone(ADMIN_PHONE):
            _send_carer(q, TIMEOUT_REPLY, "timeout")
        name = q.get("carer_name") or "a carer"
        how = (f"Reply Q{q['id']} yes, or Q{q['id']} + the right answer."
               if q["status"] == "draft" else f"Reply Q{q['id']} + your answer.")
        _text_paul(f"Q{q['id']} still waiting ({name}, {TIMEOUT_MINUTES} min): {q['question']}\n"
                   f"I've told them I'm checking with you. {how}")
    return len(rows)


def _timeout_loop() -> None:
    import time
    while True:
        try:
            check_timeouts()
        except Exception:
            logger.exception("Help desk timeout check failed")
        time.sleep(TIMEOUT_POLL_SECONDS)


def start_worker() -> None:
    """Run the timeout checker on its own thread (started by sms_orchestrator.py)."""
    threading.Thread(target=_timeout_loop, name="helpdesk-timeouts", daemon=True).start()
