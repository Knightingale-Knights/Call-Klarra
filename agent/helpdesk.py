"""
Carer help desk over SMS. Klarra sits in the middle of every conversation between Paul
and carers, and checks every answer with Paul before a carer sees it.

Carer -> Klarra (any text from a known carer not caught by another flow):
  1. The text is embedded and the 3 closest saved answers are pulled from helpdesk_kb
     (match_helpdesk_kb RPC). The recent conversation with this carer (helpdesk_messages)
     is included, so replies to Paul's relayed messages make sense.
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
approved | corrected | ack), helpdesk_messages (full carer conversation log).
If embeddings fail (e.g. OpenAI quota) it falls back to the most recent saved answers.
"""

import os
import re
import json
import logging
from datetime import datetime, timedelta, timezone

import anthropic

import db

logger = logging.getLogger("knightingale-helpdesk")

MODEL = os.environ.get("HELPDESK_MODEL", "claude-haiku-4-5").strip()
FALLBACK_MODEL = "claude-sonnet-4-6"
ADMIN_PHONE = "+61426512584"
HOLDING_REPLY = "Just a sec, working on this."
TOP_K = 3
FALLBACK_KB_ROWS = 40
HISTORY_DAYS = 7
HISTORY_MESSAGES = 8

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
             replace_id: int | None = None) -> int | None:
    client = db.get_client()
    payload = {"question": question.strip(), "answer": _clean(answer),
               "embedding": _embed(f"{question}\n{answer}")}
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


# --- carer side ------------------------------------------------------------

TRIAGE_SYSTEM = (
    "You are Klarra, the SMS assistant for Knightingale, a Melbourne aged care and NDIS "
    "nursing staffing agency run by Paul. A carer (nurse or support worker) has texted. "
    "You get the recent conversation with them and a few saved answers Paul has taught "
    "you. Paul checks every answer before the carer sees it, so draft the reply you "
    "would send. Decide kind:\n"
    "  draft = the saved answers or the conversation give you what you need to reply "
    "(questions, and also situations like being sick, running late, can't make a shift, "
    "pay or timesheet issues). Write the reply using ONLY what the saved answers and "
    "conversation say. Never guess or add facts.\n"
    "  unknown = you don't have what you need to reply. If in doubt, choose unknown.\n"
    "  ack = a bare acknowledgement with nothing to answer (ok, will do, thanks, noted).\n"
    "used_kb_id: the [number] of the saved answer you relied on, or null.\n"
    "Replies are SMS: warm, plain, brief, Australian English, no markdown, no em dashes, "
    "use their first name at most once.\n"
    'Respond ONLY with JSON: {"kind":"draft|unknown|ack","used_kb_id":number|null,'
    '"reply":"text or empty"}'
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
        decision = _json(_claude(
            TRIAGE_SYSTEM,
            f"Carer first name: {name}\n\nRecent conversation:\n{history or '(none)'}\n\n"
            f"Saved answers:\n{kb_text}\n\nCarer's new text: \"{body}\"",
            max_tokens=400,
        ))
    except Exception:
        logger.exception("Help desk triage failed, escalating")
        decision = {}

    kind = (decision.get("kind") or "unknown").lower()
    draft = _clean(decision.get("reply"))
    used = decision.get("used_kb_id")
    used = used if isinstance(used, int) and any(c.get("id") == used for c in candidates) else None

    client = db.get_client()
    row = {
        "phone": db.normalise_phone(phone),
        "nurse_id": carer.get("id"),
        "carer_name": name,
        "question": body,
        "kind": "general",
    }

    if kind == "ack":
        client.table("helpdesk_questions").insert({
            **row, "kind": "chat", "status": "ack", "answered_at": _now(),
        }).execute()
        if was_relay or test:
            _text_paul(f"{who} replied: {body}")
        return ""

    if kind == "draft" and draft:
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

    _log(phone, "out", HOLDING_REPLY, nurse_id=carer.get("id"), source="holding",
         question_id=q["id"])
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
    "no em dashes. You may say you've checked with Paul.\n"
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
                kb_id = _save_kb(out["kb_question"], out["kb_answer"], "approved_draft", qid)
                _bump_kb(kb_id, "approvals")
        client.table("helpdesk_questions").update({
            "status": "approved", "paul_answer": answer, "reply_sent": draft,
            "kb_id": kb_id, "answered_at": _now(),
        }).eq("id", qid).execute()
        return f"Sent to {name}."

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
    kb_id = _save_kb(kb_q, kb_a, "paul", qid, replace_id=replace_id)
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
