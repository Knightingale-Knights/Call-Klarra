"""
Paul's text commands for participant questionnaires (tools for admin_agent.py).

  start_my_questionnaire(participant)
      Paul answers himself. Next incomplete section, one question per text, 15 min
      limit. Anything he doesn't know is left for carers (not flagged, no email).

  create_emergency_plan(participant, redo=False)
      Write the participant's Bubble Emergency Plan from their answers, risks and
      progress notes (emergency_plans.py). Texts Paul when it's saved.

  text_carer_questionnaire(participant, carer=None)
      Find today's shift for the participant (optionally checking the carer's name),
      text the carer a heads-up now, and start the questions 30 min after the shift
      starts (straight away if that time has passed). The worker starts it once the
      carer has checked in; unanswered questions are emailed to Paul at the end.
"""

import logging
from datetime import datetime, timedelta, timezone

import db
import emergency_plans
import questionnaire as q
import questionnaire_sms as qs

logger = logging.getLogger("knightingale-agent.questionnaire-admin")

START_OFFSET = timedelta(minutes=30)


def _participants() -> list[dict]:
    return q.bubble_get("user", [{"key": "account type", "constraint_type": "equals",
                                  "value": "participant"}])


def _full_name(u: dict) -> str:
    return f"{(u.get('first name') or '').strip()} {(u.get('last name') or '').strip()}".strip()


def resolve_participant(name: str) -> tuple[dict | None, str | None]:
    """(user, None) or (None, problem text)."""
    words = [w.lower() for w in (name or "").split()]
    if not words:
        return None, "Which participant?"
    users = _participants()
    hits = [u for u in users if q.first_name(u).lower() == words[0]]
    if len(words) > 1 and len(hits) != 1:
        hits = [u for u in users if " ".join(words) in _full_name(u).lower()] or hits
    if not hits:
        return None, f"I couldn't find a participant called {name}."
    if len(hits) > 1:
        return None, ("Which one: " + ", ".join(_full_name(u) for u in hits[:5]) + "?")
    return hits[0], None


def _close_active(phone: str) -> None:
    """Close any session already in progress on this number before starting another."""
    s = qs.active_session(phone)
    if s and s.get("state"):
        qs.finish(s, s["state"])


def start_my_questionnaire(participant: str, phone: str, **_) -> dict:
    user, problem = resolve_participant(participant)
    if problem:
        return {"error": problem}
    pbid = user["_id"]
    section = None
    for sec in q.SECTION_ORDER:
        if q.missing_fields(sec, q.get_record(sec, pbid)):
            section = sec
            break
    if not section:
        return {"reply_text": f"{q.first_name(user)}'s questionnaire is all filled in. Nothing left to ask."}
    phone = db.normalise_phone(phone)
    _close_active(phone)
    session = qs.create_session(pbid, section, None, phone, None,
                                datetime.now(timezone.utc), status="Calling")
    body = qs.start(session, {"first name": "Paul"}, user, phone, admin=True, send=False)
    if not body:
        return {"reply_text": f"Nothing left to ask in {q.first_name(user)}'s {section}."}
    return {"reply_text": body}


def _todays_shifts(pbid: str) -> list[dict]:
    today = q.melb_now().date()
    lo = datetime(today.year, today.month, today.day, tzinfo=q.MELB) - timedelta(minutes=1)
    shifts = q.bubble_get("shift", [
        {"key": "participant", "constraint_type": "equals", "value": pbid},
        {"key": "date", "constraint_type": "greater than", "value": lo.isoformat()},
        {"key": "date", "constraint_type": "less than", "value": (lo + timedelta(days=1)).isoformat()},
    ])
    out = [s for s in shifts
           if str(s.get("cancelled", "")).lower() not in ("yes", "true")
           and s.get("carer") and q.shift_start(s)]
    return sorted(out, key=q.shift_start)


def text_carer_questionnaire(participant: str, carer: str | None = None, **_) -> dict:
    user, problem = resolve_participant(participant)
    if problem:
        return {"error": problem}
    pbid, p_name = user["_id"], q.first_name(user)
    shifts = _todays_shifts(pbid)
    if not shifts:
        return {"error": f"{p_name} has no shift with a carer today."}
    chosen, on_shift = None, []
    for s in shifts:
        c = q.bubble_user(s["carer"])
        on_shift.append(_full_name(c))
        if not carer or carer.split()[0].lower() == q.first_name(c).lower() \
                or carer.lower() in _full_name(c).lower():
            chosen = chosen or s
    if not chosen:
        return {"error": f"{carer} isn't on a shift with {p_name} today. On today: "
                         + ", ".join(on_shift) + "."}
    start_at = q.shift_start(chosen) + START_OFFSET
    now = q.melb_now()
    immediate = start_at <= now
    res = qs.schedule_for_shift(pbid, chosen, now if immediate else start_at, "today",
                                heads_up=not immediate)
    if res.get("error"):
        return {"error": res["error"]}
    if immediate:
        text = (f"Done. Texting {res['carer_name']} {p_name}'s {res['section']} questions "
                f"now (once they've checked in).")
    else:
        text = (f"Done. I've given {res['carer_name']} a heads-up and will text them "
                f"{p_name}'s {res['section']} questions at {q.pretty_time(start_at)}.")
    return {"reply_text": text}


def create_emergency_plan(participant: str, phone: str, redo: bool = False, **_) -> dict:
    """Write the participant's Emergency Plan in the background; Paul gets a text when done."""
    user, problem = resolve_participant(participant)
    if problem:
        return {"error": problem}
    emergency_plans.build_in_background(user, db.normalise_phone(phone), redo=bool(redo))
    p = q.first_name(user)
    what = "rewriting" if redo else "writing"
    return {"reply_text": f"On it. I'm {what} {p}'s emergency plan now and will text you when "
                          f"it's in Bubble (about a minute)."}
