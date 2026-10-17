"""
Carer help desk over SMS.

Every text from a known carer (a row in `nurses`) that isn't part of another flow
(questionnaire, shift offer, check-in/out link) lands here:

  1. The text is embedded (OpenAI text-embedding-3-small) and the 3 closest saved
     answers are pulled from `helpdesk_kb` via the match_helpdesk_kb RPC.
  2. Claude (fast model) reads just those 3 and decides:
       answered - the saved answers cover it -> reply straight away
       chat     - small talk / acknowledgement -> short friendly reply, no escalation
       unknown  - a general question we have no answer for yet
       personal - about their own situation (sick, late, pay, a specific shift or
                  participant) or a message meant for Paul
  3. unknown / personal: the carer gets "Just a sec, working on this." and Paul gets
     "Q12 from Kelly: ...". Paul replies "Q12 <answer>". Klarra turns that into a reply
     for the carer, texts it, and (for general questions only) saves the Q&A to
     helpdesk_kb with an embedding, so the next carer asking the same thing, in any
     wording, gets answered automatically.

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


# --- carer side ---

TRIAGE_SYSTEM = (
    "You run the SMS help desk for Knightingale, a Melbourne aged care and NDIS nursing "
    "staffing agency run by Paul. A carer (nurse or support worker) has texted. You are "
    "given a few saved answers from the help desk knowledge base. Decide ONE kind:\n"
    "answered = the saved answers clearly cover what they asked. Write the reply using "
    "ONLY facts in the saved answers. Never guess or add anything.\n"
    "chat = small talk or an acknowledgement with nothing to answer (ok, sounds good, "
    "good morning). Write a very short friendly reply.\n"
    "unknown = a general question about working at Knightingale (policies, pay rules, "
    "uniforms, procedures, the app, facilities) that the saved answers do not fully cover.\n"
    "personal = about their own situation that Paul needs to handle (sick, running late, "
    "can't make a shift, swapping shifts, their own pay or timesheet, an incident, a "
    "specific participant), or a message meant for Paul.\n"
    "If in doubt between answered and unknown, choose unknown.\n"
    "Replies are SMS: warm, plain, brief, Australian English, no markdown, no em dashes, "
    "use their first name at most once.\n"
    'Respond ONLY with JSON: {"kind":"answered|chat|unknown|personal","reply":"text or empty"}'
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

    candidates = _candidates(body)
    kb_text = "\n\n".join(
        f"Q: {c['question']}\nA: {c['answer']}" for c in candidates
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
    reply = (decision.get("reply") or "").replace("—", ",").strip()

    client = db.get_client()
    row = {
        "phone": db.normalise_phone(phone),
        "nurse_id": carer.get("id"),
        "carer_name": name,
        "question": body,
    }

    if kind in ("answered", "chat") and reply:
        client.table("helpdesk_questions").insert({
            **row, "kind": "general" if kind == "answered" else "chat",
            "status": "auto_answered", "reply_sent": reply, "answered_at": _now(),
        }).execute()
        return reply

    kind = "personal" if kind == "personal" else "general"
    q = client.table("helpdesk_questions").insert({**row, "kind": kind}).execute().data[0]
    who = f"{name} {carer.get('last_name') or ''}".strip()
    if test:
        who += " (test)"
    try:
        db.send_sms(ADMIN_PHONE, f"Q{q['id']} from {who}:\n{body}\n\nReply Q{q['id']} + your answer.")
    except Exception:
        logger.exception("Failed to text Paul help desk question Q%s", q["id"])
    return HOLDING_REPLY


# --- Paul side ---

COMPOSE_SYSTEM = (
    "You help Paul, director of Knightingale (a Melbourne aged care and NDIS staffing "
    "agency), answer a carer's text. You get the carer's message and Paul's answer, which "
    "may be terse. Write:\n"
    "reply = the SMS to send the carer. Use ONLY what Paul said; never add facts, "
    "conditions or advice. Warm, plain, brief, Australian English, no markdown, no em "
    "dashes. Don't sign off as Paul and don't mention Paul unless his answer does.\n"
    "kb_question = the carer's question rewritten as a general question anyone could ask "
    "(no names or dates specific to this carer).\n"
    "kb_answer = Paul's answer as a general, standalone answer to kb_question, again "
    "using only what Paul said.\n"
    'Respond ONLY with JSON: {"reply":"...","kb_question":"...","kb_answer":"..."}'
)


def handle_paul_answer(body: str) -> str | None:
    """Paul's 'Q12 <answer>' text. Returns the confirmation reply for Paul, or None if
    the text isn't a help desk answer (caller carries on)."""
    m = ANSWER_PATTERN.match(body or "")
    if not m:
        return None
    qid, answer = int(m.group(1)), m.group(2).strip()

    client = db.get_client()
    r = client.table("helpdesk_questions").select("*").eq("id", qid).limit(1).execute()
    if not r.data:
        return None  # not a help desk question number: let normal routing have it
    q = r.data[0]
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
        logger.exception("Help desk compose failed, sending Paul's answer as written")
        out = {}

    reply = (out.get("reply") or answer).replace("—", ",").strip()

    try:
        db.send_sms(q["phone"], reply)
    except Exception:
        logger.exception("Failed to text help desk answer for Q%s", qid)
        return f"Couldn't text {q.get('carer_name') or 'the carer'} just now. Try Q{qid} again in a minute."

    kb_id = None
    if q.get("kind") == "general":
        kb_q = (out.get("kb_question") or q["question"]).strip()
        kb_a = (out.get("kb_answer") or answer).strip()
        try:
            kb = client.table("helpdesk_kb").insert({
                "question": kb_q, "answer": kb_a, "source": "paul",
                "source_question_id": qid,
                "embedding": _embed(f"{kb_q}\n{kb_a}"),
            }).execute().data[0]
            kb_id = kb["id"]
        except Exception:
            logger.exception("Failed to save help desk answer for Q%s", qid)

    client.table("helpdesk_questions").update({
        "status": "answered", "paul_answer": answer, "reply_sent": reply,
        "kb_id": kb_id, "answered_at": _now(),
    }).eq("id", qid).execute()

    name = q.get("carer_name") or "the carer"
    return f"Sent to {name}." + (" Saved for next time." if kb_id else "")
