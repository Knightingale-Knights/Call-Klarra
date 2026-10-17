"""
Carer help desk over SMS. Klarra is always the intermediary between carers and Paul.

Every text from a known carer (a row in `nurses`) that isn't part of another flow
(questionnaire, shift offer, check-in/out link) lands here:

  1. The text is embedded (OpenAI text-embedding-3-small) and the 3 closest saved
     answers are pulled from `helpdesk_kb` via the match_helpdesk_kb RPC.
  2. Claude (fast model) reads just those 3 and decides:
       answered - the saved answers cover it -> Klarra replies straight away
       chat     - small talk / acknowledgement -> short friendly reply
       unknown  - nothing saved covers it yet
     plus notify_paul for things Paul needs to know about (sick, late, pay, incident).
     While she's learning (HELPDESK_REVIEW_ALL, on by default) Paul is texted every
     answer she gives: "Q12 I answered Kelly: They said... I replied...". He replies
     "Q12 good", or "Q12 <right answer>": she learns the correction (replacing the
     saved answer if it was wrong) and texts the carer a correction.
  3. unknown: the carer gets "Just a sec, working on this." and Paul gets
     "Q12 from Kelly: ...". Paul replies "Q12 <answer>" to Klarra. Klarra learns it
     (saves a general version of the Q&A to helpdesk_kb, embedded), then texts the
     carer the reply herself. Next time anyone asks the same thing, in any wording,
     she answers on her own (and still FYIs Paul if it's a notify-type message).

Paul can test it from his own number by starting a text with "?" ("? what's the uniform").

Tables (Supabase): helpdesk_kb, helpdesk_questions. RPC: match_helpdesk_kb.
If embeddings fail (e.g. OpenAI quota), it falls back to sending the most recent
saved answers instead of the top 3, so it keeps working, just less efficiently.
"""

import os
import re
import json
import logging

import anthropic

import db

logger = logging.getLogger("knightingale-helpdesk")

MODEL = os.environ.get("HELPDESK_MODEL", "claude-haiku-4-5").strip()
FALLBACK_MODEL = "claude-sonnet-4-6"
ADMIN_PHONE = "+61426512584"
HOLDING_REPLY = "Just a sec, working on this."
TOP_K = 3
# While Klarra is learning, Paul is texted every answer she gives so he can approve or
# correct it. Set HELPDESK_REVIEW_ALL=0 later to only hear about notify-type messages.
REVIEW_ALL = os.environ.get("HELPDESK_REVIEW_ALL", "1").strip() != "0"
GOOD_WORDS = {"good", "great", "yes", "y", "correct", "ok", "okay", "perfect", "spot on",
              "good answer", "all good", "thats right", "that's right", "right", "\U0001F44D"}
FALLBACK_KB_ROWS = 40

ANSWER_PATTERN = re.compile(r"^\s*q\s*(\d+)\b[\s:.\-]*(.*)$", re.IGNORECASE | re.DOTALL)


# --- helpers ---

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
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group(0)) if m else {}


def _clean(text: str) -> str:
    return (text or "").replace("—", ",").replace("–", "-").strip()


def _embed(text: str) -> list[float] | None:
    try:
        return db.embed_situation(text)
    except Exception:
        logger.exception("Help desk embedding failed, using fallback KB list")
        return None


def _backfill_embeddings() -> None:
    """Embed any saved answers that don't have an embedding yet (e.g. rows seeded
    from Virtual Paul, or rows saved while OpenAI was down)."""
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


def carer_for(phone: str) -> dict | None:
    """The nurses row for this number, or None if it isn't a known carer."""
    phone = db.normalise_phone(phone)
    if not phone:
        return None
    r = (db.get_client().table("nurses").select("id, first_name, last_name, phone")
         .eq("phone", phone).limit(1).execute())
    return r.data[0] if r.data else None


def _text_paul(body: str) -> None:
    try:
        db.send_sms(ADMIN_PHONE, body)
    except Exception:
        logger.exception("Failed to text Paul from help desk")


# --- carer side ---

TRIAGE_SYSTEM = (
    "You are Klarra, the SMS help desk for Knightingale, a Melbourne aged care and NDIS "
    "nursing staffing agency run by Paul. A carer (nurse or support worker) has texted. "
    "You are given a few saved answers Paul has taught you. Decide:\n"
    "kind:\n"
    "  answered = the saved answers clearly cover this kind of message (questions, and "
    "also situations like being sick, running late, can't make a shift, pay or timesheet "
    "issues). Write the reply using ONLY what the saved answers say. Never guess or add "
    "anything.\n"
    "  chat = small talk or an acknowledgement with nothing to answer (ok, sounds good, "
    "good morning). Write a very short friendly reply.\n"
    "  unknown = the saved answers do not fully cover it. If in doubt, choose unknown.\n"
    "notify_paul: true if Paul needs to know about this message even when you can answer "
    "it (sick, running late, can't make or wants to swap a shift, pay or timesheet "
    "problem, an incident or injury, anything about a participant's wellbeing, a "
    "complaint). false for general questions and chat.\n"
    "used_kb_id: the [number] of the saved answer you relied on, or null.\n"
    "Replies are SMS: warm, plain, brief, Australian English, no markdown, no em dashes, "
    "use their first name at most once.\n"
    'Respond ONLY with JSON: {"kind":"answered|chat|unknown","notify_paul":true|false,'
    '"used_kb_id":number|null,"reply":"text or empty"}'
)


def handle_carer_text(phone: str, body: str, carer: dict | None = None,
                      test: bool = False) -> str | None:
    """Answer a carer's text, or escalate it to Paul. Returns the SMS reply for the
    carer, or None if this isn't a known carer (caller carries on)."""
    body = (body or "").strip()
    if not body:
        return None
    carer = carer or carer_for(phone)
    if not carer:
        return None
    name = carer.get("first_name") or "Carer"
    who = f"{name} {carer.get('last_name') or ''}".strip() + (" (test)" if test else "")

    candidates = _candidates(body)
    kb_text = "\n\n".join(
        f"[{c['id']}] Q: {c['question']}\nA: {c['answer']}" for c in candidates
    ) or "(no saved answers yet)"

    try:
        decision = _json(_claude(
            TRIAGE_SYSTEM,
            f"Carer first name: {name}\n\nSaved answers:\n{kb_text}\n\nCarer's text: \"{body}\"",
            max_tokens=400,
        ))
    except Exception:
        logger.exception("Help desk triage failed, escalating")
        decision = {}

    kind = (decision.get("kind") or "unknown").lower()
    notify = bool(decision.get("notify_paul"))
    reply = _clean(decision.get("reply"))

    client = db.get_client()
    row = {
        "phone": db.normalise_phone(phone),
        "nurse_id": carer.get("id"),
        "carer_name": name,
        "question": body,
        "kind": "notify" if notify else ("chat" if kind == "chat" else "general"),
    }

    if kind in ("answered", "chat") and reply:
        used = decision.get("used_kb_id")
        used = used if isinstance(used, int) and any(c.get("id") == used for c in candidates) else None
        q = client.table("helpdesk_questions").insert({
            **row, "status": "auto_answered", "reply_sent": reply, "answered_at": _now(),
            "kb_id": used,
        }).execute().data[0]
        if kind == "answered" and (REVIEW_ALL or notify):
            label = "FYI, I answered" if notify else "I answered"
            _text_paul(
                f"Q{q['id']} {label} {who}:\nThey said: {body}\nI replied: {reply}\n\n"
                f"Reply Q{q['id']} good, or Q{q['id']} + the right answer."
            )
        return reply

    q = client.table("helpdesk_questions").insert(row).execute().data[0]
    _text_paul(f"Q{q['id']} from {who}:\n{body}\n\nReply Q{q['id']} + your answer.")
    return HOLDING_REPLY


# --- Paul side ---

COMPOSE_SYSTEM = (
    "You are Klarra, the SMS assistant for Knightingale (a Melbourne aged care and NDIS "
    "staffing agency). A carer texted, and Paul (the director) has told you how to "
    "respond. His answer may be terse. Write:\n"
    "reply = the SMS you, Klarra, send the carer. Use ONLY what Paul said; never add "
    "facts, conditions or advice. Warm, plain, brief, Australian English, no markdown, "
    "no em dashes. You may say you've checked with Paul.\n"
    "kb_question = the carer's message rewritten as a general situation or question any "
    "carer could send (no names, dates or details specific to this carer), e.g. "
    "\"What should I do if I'm sick and can't make my shift?\".\n"
    "kb_answer = how to respond to kb_question in future, as a general standalone answer, "
    "using only what Paul said (drop anything only true for this carer this time).\n"
    'Respond ONLY with JSON: {"reply":"...","kb_question":"...","kb_answer":"..."}'
)


def handle_paul_answer(body: str) -> str | None:
    """Paul's 'Q12 <answer>' text. Klarra learns from it, then texts the carer.
    Returns the confirmation reply for Paul, or None if the text isn't a help desk
    answer (caller carries on)."""
    m = ANSWER_PATTERN.match(body or "")
    if not m:
        return None
    qid, answer = int(m.group(1)), m.group(2).strip()

    client = db.get_client()
    r = client.table("helpdesk_questions").select("*").eq("id", qid).limit(1).execute()
    if not r.data:
        return None  # not a help desk question number: let normal routing have it
    q = r.data[0]
    if q["status"] == "auto_answered":
        return _review_auto_answer(q, answer)
    if q["status"] != "pending":
        return f"Q{qid} has already been answered."
    if not answer:
        return f"Add your answer after Q{qid}, e.g. Q{qid} yes that's fine."

    try:
        out = _json(_claude(
            COMPOSE_SYSTEM,
            f"Carer: {q.get('carer_name')}\nCarer's message: \"{q['question']}\"\n\n"
            f"Paul's answer: \"{answer}\"",
            max_tokens=500,
        ))
    except Exception:
        logger.exception("Help desk compose failed, using Paul's answer as written")
        out = {}

    # Learn first, so the answer is saved even if the text to the carer fails.
    kb_id = q.get("kb_id")  # already learnt on an earlier attempt whose text failed
    kb_q = (out.get("kb_question") or q["question"]).strip()
    kb_a = _clean(out.get("kb_answer") or answer)
    try:
        payload = {"question": kb_q, "answer": kb_a, "embedding": _embed(f"{kb_q}\n{kb_a}")}
        if kb_id:
            client.table("helpdesk_kb").update(payload).eq("id", kb_id).execute()
        else:
            kb = client.table("helpdesk_kb").insert({
                **payload, "source": "paul", "source_question_id": qid,
            }).execute().data[0]
            kb_id = kb["id"]
    except Exception:
        logger.exception("Failed to save help desk answer for Q%s", qid)

    reply = _clean(out.get("reply") or answer)
    name = q.get("carer_name") or "the carer"
    try:
        db.send_sms(q["phone"], reply)
    except Exception:
        logger.exception("Failed to text help desk answer for Q%s", qid)
        client.table("helpdesk_questions").update({"kb_id": kb_id}).eq("id", qid).execute()
        return f"Learnt it, but couldn't text {name} just now. Try Q{qid} again in a minute."

    client.table("helpdesk_questions").update({
        "status": "answered", "paul_answer": answer, "reply_sent": reply,
        "kb_id": kb_id, "answered_at": _now(),
    }).eq("id", qid).execute()

    return (f"Learnt it and texted {name}." if kb_id else f"Texted {name}.") + f"\nSent: {reply}"


# --- Paul reviewing an answer Klarra gave on her own ---

CORRECT_SYSTEM = (
    "You are Klarra, the SMS assistant for Knightingale (a Melbourne aged care and NDIS "
    "staffing agency). You answered a carer's text on your own using a saved answer, and "
    "Paul (the director) has corrected you. His correction may be terse. Write:\n"
    "reply = a follow-up SMS to the carer correcting your earlier reply, e.g. starting "
    "\"Quick update on my last message:\". Use ONLY what Paul said. Warm, plain, brief, "
    "Australian English, no markdown, no em dashes.\n"
    "kb_question = the carer's message as a general situation or question any carer could "
    "send (nothing specific to this carer).\n"
    "kb_answer = how to respond to kb_question in future, general and standalone, using "
    "only what Paul said.\n"
    "saved_answer_wrong = true if the saved answer itself is wrong according to Paul (so it "
    "should be replaced), false if the saved answer is fine but just didn't fit this "
    "message.\n"
    'Respond ONLY with JSON: {"reply":"...","kb_question":"...","kb_answer":"...",'
    '"saved_answer_wrong":true|false}'
)


def _review_auto_answer(q: dict, verdict: str) -> str:
    client = db.get_client()
    qid = q["id"]
    name = q.get("carer_name") or "the carer"
    if not verdict:
        return f"Reply Q{qid} good, or Q{qid} + the right answer."

    if verdict.lower().strip(" !.") in GOOD_WORDS:
        client.table("helpdesk_questions").update(
            {"status": "approved", "paul_answer": verdict}).eq("id", qid).execute()
        return f"Thanks, noted Q{qid} as a good answer."

    saved = None
    if q.get("kb_id"):
        r = client.table("helpdesk_kb").select("id, question, answer").eq("id", q["kb_id"]).limit(1).execute()
        saved = r.data[0] if r.data else None

    try:
        out = _json(_claude(
            CORRECT_SYSTEM,
            f"Carer: {name}\nCarer's message: \"{q['question']}\"\n"
            f"Your reply: \"{q.get('reply_sent')}\"\n"
            f"Saved answer you used: "
            f"{json.dumps(saved) if saved else 'none'}\n\n"
            f"Paul's correction: \"{verdict}\"",
            max_tokens=500,
        ))
    except Exception:
        logger.exception("Help desk correction compose failed, using Paul's words")
        out = {}

    kb_q = (out.get("kb_question") or q["question"]).strip()
    kb_a = _clean(out.get("kb_answer") or verdict)
    payload = {"question": kb_q, "answer": kb_a, "embedding": _embed(f"{kb_q}\n{kb_a}")}
    kb_id, replaced = None, False
    try:
        if saved and out.get("saved_answer_wrong"):
            client.table("helpdesk_kb").update(payload).eq("id", saved["id"]).execute()
            kb_id, replaced = saved["id"], True
        else:
            kb_id = client.table("helpdesk_kb").insert({
                **payload, "source": "paul", "source_question_id": qid,
            }).execute().data[0]["id"]
    except Exception:
        logger.exception("Failed to save help desk correction for Q%s", qid)

    reply = _clean(out.get("reply") or f"Quick update on my last message: {verdict}")
    try:
        db.send_sms(q["phone"], reply)
        sent = True
    except Exception:
        logger.exception("Failed to text help desk correction for Q%s", qid)
        sent = False

    client.table("helpdesk_questions").update({
        "status": "corrected", "paul_answer": verdict, "kb_id": kb_id,
        "reply_sent": reply if sent else q.get("reply_sent"),
    }).eq("id", qid).execute()

    learnt = "Updated what I knew" if replaced else "Learnt it"
    if not kb_id:
        learnt = "Couldn't save it"
    if not sent:
        return f"{learnt}, but couldn't text {name} the correction."
    return f"{learnt} and texted {name} the correction.\nSent: {reply}"
