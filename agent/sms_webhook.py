"""
SMS webhook — receives a facility's text, parses the shift request, logs it, replies.
The orchestrator picks it up and (for sms) texts the facility the result.

Also routes:
  - Two ad hoc admin commands from Paul's number: an availability query ("who can
    do EN AM Wednesday") and a targeted single-nurse text ("text Maria for the PM
    shift Thursday at Port Melbourne"), independent of the ranked cascade and of any
    shift_request.
  - A nurse replying YES/NO to an active offer (ranked cascade, or an ad hoc
    single-nurse offer) -> updates sms_nurse_offers / sms_adhoc_offers.
  - Paul replying OK to an admin-approval request -> updates sms_shift_state.

Routing order matters, and is deliberately this order:
  0. Paul's ad hoc admin commands (availability query / text a specific nurse) —
     checked first on his number, ahead of a pending approval, since they're a
     distinct explicit intent and shouldn't get swallowed by the approval gate.
  1. Admin approval reply (narrow: only fires if there's a real pending approval).
  2. An ACTIVE offer (cascade or ad hoc) on this number — wins over facility
     identity, because it means we are actively expecting a YES/NO from this exact
     number right now. This matters because in dev/test setups one phone number can
     simultaneously be a registered facility AND the stand-in nurse (both sharing
     KLARRA_DEV_PHONE) — without this priority, a nurse's genuine YES/NO gets
     swallowed by the facility-request parser instead of reaching the offer.
  3. Recognised facility -> shift-request parsing.
  4. Not a facility, but has past cascade-offer history -> a late YES still claims
     the shift if nobody else has, otherwise they're told another carer got it.
     (A late ad hoc reply is already caught by step 2 — an ad hoc offer stays
     'active' until answered.)
  5. Otherwise -> afterhours chat / unrecognised-number fallback.

Run:  python agent/sms_webhook.py
"""

import os
import json
import random
import logging

import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from dotenv import load_dotenv
from flask import Flask, request, Response
import anthropic

import db

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("knightingale-sms")

app = Flask(__name__)

# Roles Knightingale staffs. Must match the values sync_bubble.py writes to
# nurses.role — get_candidate_pool matches on this string exactly, so a request
# parsed as a role no carer holds finds an empty pool.
STAFFED_ROLES = ["RN", "EN", "PCA", "DSW"]

ACK_REPLIES = [
    "Absolutely, working on this — one moment.",
    "Sure, we're on it.",
    "Not a problem. Just a moment please.",
    "Got it — on it now.",
    "Of course, leave it with me a sec.",
]


def _today_melb() -> str:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("Australia/Melbourne")).strftime("%Y-%m-%d")


def parse_request(text: str) -> dict | None:
    """Use Claude to pull date/shift/role/times/facility from the text. Returns dict or None.

    The role list here must stay in step with STAFFED_ROLES. It previously offered only
    EN|RN, which meant a PCA request was silently coerced to the nearest allowed value
    and sent to the wrong carers — so the prompt now names every staffed role and is
    told to error rather than guess when the role is something else."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": (
                "Extract a care shift request from this SMS. Knightingale staffs four roles: "
                "RN (registered nurse), EN (enrolled nurse), PCA (personal care assistant — "
                "also called AIN, PCW, or care worker), and DSW (disability support worker — "
                "also called support worker). Respond ONLY with JSON: "
                '{"date":"YYYY-MM-DD","shift_type":"Morning|Afternoon|Night",'
                '"role":"RN|EN|PCA|DSW",'
                '"start_time":"HH:MM or null","end_time":"HH:MM or null",'
                '"facility":"site name as written in the SMS, or null if none is mentioned"}. '
                "Use the role the SMS actually asks for. Never substitute a different role "
                "because one seems close — if the requested role is not one of the four "
                "above, or no role is stated, respond {\"error\":\"role\"}. "
                "start_time/end_time are 24-hour HH:MM if the SMS states or clearly implies "
                "specific times (e.g. '2-10pm', 'from 14:00'), otherwise null — do not guess. "
                "If you cannot determine date or shift_type, respond {\"error\":\"...\"}. "
                f"Today is {_today_melb()} (Australia/Melbourne). SMS: \"{text}\""
            ),
        }],
    )
    raw = msg.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(raw)
    except Exception:
        return None
    if "error" in data:
        return None
    # Belt and braces: never log a request for a role we don't staff, even if the
    # model returns one anyway.
    role = str(data.get("role") or "").strip().upper()
    if role not in STAFFED_ROLES:
        logger.warning("Parsed unstaffed role %r from SMS: %s", data.get("role"), text)
        return None
    data["role"] = role
    return data


def parse_admin_command(text: str) -> dict:
    """
    Classify a text from Paul (the admin number) as one of two ad hoc commands, or
    neither. Both are separate from the normal facility shift-request flow and from
    the ranked SMS cascade — Command A just answers a question, Command B texts
    exactly one named carer with no ranking or shift_request involved.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": (
                "This text is from Paul, the admin of Knightingale (an aged-care/NDIS "
                "staffing agency), to Klarra, his scheduling assistant. Decide which of "
                "two specific commands it is, or neither.\n\n"
                "Command A - AVAILABILITY QUERY: a question asking which carers of a "
                "role can do a shift. No specific carer is named. Roles: RN, EN, PCA, "
                "DSW. Shift blocks: AM=Morning, PM=Afternoon, NS=Night. Respond ONLY: "
                '{"type":"availability","role":"RN|EN|PCA|DSW",'
                '"shift_type":"Morning|Afternoon|Night","date":"YYYY-MM-DD"}\n\n'
                "Command B - TEXT A SPECIFIC NURSE: asks to text ONE named carer (a "
                "person's first name or full name appears) about a shift. Respond "
                'ONLY: {"type":"text_nurse","nurse_name":"as written",'
                '"facility":"site name as written, or null if not mentioned",'
                '"date":"YYYY-MM-DD","shift_type":"Morning|Afternoon|Night"}\n\n'
                "If it's a normal shift request reporting a role/date/site that needs "
                "covering — no named carer, not phrased as a question — or anything "
                "else, respond ONLY {\"type\":\"none\"}.\n\n"
                "Only use a YYYY-MM-DD date you can actually resolve from the message; "
                "if the date is unclear, respond {\"type\":\"none\"}.\n\n"
                f"Today is {_today_melb()} (Australia/Melbourne). Message: \"{text}\""
            ),
        }],
    )
    raw = msg.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"type": "none"}


def twiml_reply(text: str) -> Response:
    body = f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{text}</Message></Response>'
    return Response(body, mimetype="text/xml")


def _notify_facility_sms(req: dict, filled: bool, nurse_name: str | None):
    """Text the facility the result of an approved shift."""
    to = req.get("facility_callback_number")
    if not to:
        return
    if filled:
        body = (f"Good news — {nurse_name} is covering the "
                f"{req['shift_type'].lower()} shift on {db.pretty_date(req['date'])}.")
    else:
        body = (f"Sorry, no one was available for the {req['shift_type'].lower()} "
                f"shift on {db.pretty_date(req['date'])} yet. We'll keep trying.")
    try:
        db.send_sms(to, body)
    except Exception:
        logger.exception("Failed to send facility result SMS")


def _afterhours_now() -> bool:
    """True if 9pm–5am Melbourne, or dev (so it's always testable)."""
    if db.DEV:
        return True
    from datetime import datetime
    from zoneinfo import ZoneInfo
    h = datetime.now(ZoneInfo("Australia/Melbourne")).hour
    return h >= 21 or h < 5


ADMIN_PHONE = "+61426512584"

AFTERHOURS_INTRO = (
    "In order for Paul and Vidhu to do their best for you during the day, they need to "
    "sleep at night. So they've entrusted after-hours messages to me, their AI assistant. "
    "My name is Klarra. If you let me know what's going on, I'll pass it on to Vidhu and Paul."
)

AFTERHOURS_SYSTEM = (
    "You are Klarra, Knightingale's friendly after-hours AI assistant, texting with someone "
    "whose number isn't a recognised facility. Knightingale is a Melbourne aged-care/NDIS "
    "nursing staffing agency; Paul and Vidhu run it. Be warm, natural, easy-going, concise — "
    "this is SMS. Gather what's going on: who they are, the issue, which facility/person, and "
    "anything Paul and Vidhu need to act in the morning. Ask one thing at a time. When you have "
    "enough, sign off warmly, e.g. that Paul and Vidhu start around 5am and will get back to "
    "them then. When you have gathered enough and are signing off, end your message with the "
    "exact token [DONE] on its own at the very end (the user won't see it)."
)


def handle_afterhours_chat(phone: str, body: str) -> Response:
    """Run a multi-turn SMS conversation with an unknown afterhours caller."""
    thread = db.get_afterhours_thread(phone)
    messages = (thread or {}).get("messages", []) if thread else []

    # Brand-new thread: send the intro, seed history, wait for their reply.
    if not messages:
        messages = [{"role": "assistant", "content": AFTERHOURS_INTRO}]
        db.save_afterhours_thread(phone, messages)
        return twiml_reply(AFTERHOURS_INTRO)

    # If a prior thread was already wrapped up, start fresh on a new message.
    if thread and thread.get("done"):
        messages = [{"role": "assistant", "content": AFTERHOURS_INTRO}]
        db.save_afterhours_thread(phone, messages)
        return twiml_reply(AFTERHOURS_INTRO)

    messages.append({"role": "user", "content": body})

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        system=AFTERHOURS_SYSTEM,
        messages=messages,
    )
    reply = "".join(b.text for b in resp.content if b.type == "text").strip()

    done = "[DONE]" in reply
    reply = reply.replace("[DONE]", "").strip()
    messages.append({"role": "assistant", "content": reply})
    db.save_afterhours_thread(phone, messages, done=done)

    if done:
        _summarise_to_admin(phone, messages)

    return twiml_reply(reply)


def _summarise_to_admin(phone: str, messages: list):
    """Text Paul a summary of the after-hours conversation."""
    try:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": (
                "Summarise this after-hours enquiry for Paul in 2-4 short lines: who texted "
                "(number below), what they need, and anything urgent. Plain text.\n\n"
                f"Number: {phone}\n\n{transcript}"
            )}],
        )
        summary = "".join(b.text for b in resp.content if b.type == "text").strip()
        db.send_sms(ADMIN_PHONE, f"After-hours enquiry ({phone}):\n{summary}")
        t = db.get_afterhours_thread(phone)
        if t:
            db.save_afterhours_thread(phone, t["messages"], done=True, summarised=True)
    except Exception:
        logger.exception("Failed to summarise after-hours chat")


# --- SMS nurse-offer workflow routing ---

def _is_admin_number(phone: str) -> bool:
    admin = os.environ.get("KLARRA_DEV_PHONE", "").strip().strip('"').strip("'")
    phone = (phone or "").strip()
    return bool(admin) and phone == admin


def _parse_yes_no(text: str) -> str | None:
    t = text.strip().lower().strip("!.")
    if t in ("yes", "y", "yeah", "yep", "sure", "ok", "okay", "yup"):
        return "yes"
    if t in ("no", "n", "nah", "nope"):
        return "no"
    return None


def handle_admin_reply(body: str) -> Response | None:
    """If Paul has a pending approval and this looks like an OK, confirm it and
    return the reply. Returns None if there's nothing to approve (caller should
    fall through to normal handling)."""
    approval = db.get_pending_admin_approval()
    if not approval:
        return None
    if "ok" in body.strip().lower():
        db.mark_sms_state(approval["shift_request_id"], "confirmed", admin_approved_at="now()")
        return twiml_reply("Confirmed — texting the facility now.")
    # There's a pending approval but this reply doesn't read as an OK. Don't
    # silently swallow it, but don't mis-fire the confirm either.
    return twiml_reply("Reply OK to confirm the shift, or I'll keep waiting.")


def handle_availability_query(parsed: dict) -> Response:
    """Answer Paul's "who can do X" ad hoc query — a plain list, not a ranked pool,
    and not tied to any facility or shift_request."""
    role = str(parsed.get("role") or "").strip().upper()
    shift_type = parsed.get("shift_type")
    date = parsed.get("date")
    if role not in STAFFED_ROLES or not shift_type or not date:
        return twiml_reply(
            "Sorry, I need a role (RN/EN/PCA/DSW), a shift (AM/PM/NS) and a date to "
            "check availability."
        )
    nurses = db.get_available_nurses(date, shift_type, role)
    if not nurses:
        return twiml_reply(f"No {role} carers available {shift_type} on {db.pretty_date(date)}.")
    names = ", ".join(n["first_name"] for n in nurses)
    return twiml_reply(f"{role} available {shift_type} {db.pretty_date(date)}: {names}")


def adhoc_offer_message(nurse: dict, facility: dict, date: str, shift_type: str) -> str:
    return (
        f"Hi {nurse['first_name']}, I've got a shift at {facility['name']} "
        f"on {db.short_date(date)} ({shift_type}). Please reply YES if you would "
        f"like it. Please reply NO if you would prefer to pass. Thank you"
    )


def handle_text_nurse_command(parsed: dict) -> Response:
    """Text exactly one named carer about one shift. No ranking, no shift_request —
    db.send_sms already respects KLARRA_MODE, so this is blocked in mid (nurses are
    never contacted in mid) and redirected to the dev phone in dev, same as everywhere
    else."""
    name = parsed.get("nurse_name")
    date = parsed.get("date")
    shift_type = parsed.get("shift_type")
    facility_name = parsed.get("facility")

    if not (name and date and shift_type):
        return twiml_reply("Sorry, I need the carer's name, a date and a shift (AM/PM/NS).")

    nurse, candidates = db.find_nurse_by_name(name)
    if candidates:
        opts = ", ".join(f"{c['first_name']} {c['last_name']}" for c in candidates)
        return twiml_reply(f"A few carers match '{name}': {opts}. Which one?")
    if not nurse:
        return twiml_reply(f"Couldn't find a carer named '{name}'.")

    facility = db.find_facility_by_name(facility_name) if facility_name else None
    if not facility:
        return twiml_reply("Which site is this shift at?")

    msg = adhoc_offer_message(nurse, facility, date, shift_type)
    db.create_adhoc_offer(
        nurse_id=nurse["id"], facility_id=facility["id"], facility_name=facility["name"],
        date=date, shift_type=shift_type, message=msg,
    )
    db.send_sms(nurse["phone"], msg)
    return twiml_reply(
        f"Texted {nurse['first_name']} about the {shift_type} shift at "
        f"{facility['name']} on {db.pretty_date(date)}. I'll let you know."
    )


def _too_late_reply(offer: dict) -> str:
    """Told to a nurse whose YES arrived after someone else claimed the shift."""
    name = offer.get("nurse_first_name") or "there"
    role = (offer.get("shift_requests") or {}).get("role") or "carer"
    return (f"Sorry {name}, I had another {role} reply and accept the shift. "
            f"Will do my best to get you on the next one!")


def handle_offer_reply(offer: dict, body: str) -> Response:
    """A nurse replying to a shift offer.

    Offers go out every ~40s, but people reply on their own schedule — a YES five
    minutes later is completely normal behaviour, and used to be met with 'that's
    already been sorted' even when the shift was still wide open. So a YES is honoured
    whenever the shift is unclaimed, regardless of whether this nurse's own offer
    window has closed; db.claim_shift settles ties atomically, first reply wins.
    """
    answer = _parse_yes_no(body)
    req = offer.get("shift_requests") or {}
    request_id = req.get("id") or offer.get("shift_request_id")

    if answer == "yes":
        if request_id and db.claim_shift(request_id, offer["nurse_id"], offer["id"]):
            db.skip_remaining_offers(request_id, except_offer_id=offer["id"])
            return twiml_reply("Great, thanks! You'll see it in the app shortly.")
        return twiml_reply(_too_late_reply(offer))

    if answer == "no":
        if offer.get("status") in ("offered", "alerted"):
            db.mark_offer(offer["id"], "declined", replied_at="now()")
        return twiml_reply("No worries, thanks for letting us know.")

    if offer.get("status") in ("offered", "alerted"):
        return twiml_reply("Sorry, I didn't catch that — reply YES or NO for the shift.")
    return None


def handle_late_nurse_reply(phone: str, body: str) -> Response | None:
    """A known nurse (has offer history) replying with no active offer right now, and
    this number isn't a recognised facility either. Their most recent offer is still
    worth honouring if that shift hasn't been claimed yet — see handle_offer_reply.
    Returns None if this phone has never received an SMS offer at all, or the reply
    isn't a yes/no (caller should fall through)."""
    latest = db.get_latest_offer_by_phone(phone)
    if not latest:
        return None
    return handle_offer_reply(latest, body)


def handle_adhoc_offer_reply(offer: dict, body: str) -> Response:
    """A nurse replying to a targeted single-nurse offer. Unlike the cascade, there's
    no shift to lose to someone else — just record the answer and tell Paul."""
    answer = _parse_yes_no(body)
    nurse_name = (offer.get("nurses") or {}).get("first_name") or "Carer"
    if answer not in ("yes", "no"):
        return twiml_reply("Sorry, I didn't catch that — reply YES or NO for the shift.")

    status = "accepted" if answer == "yes" else "declined"
    db.mark_adhoc_offer(offer["id"], status, replied_at="now()")

    late_note = (" (this came in after I'd flagged no response)"
                 if offer.get("timeout_alerted") else "")
    admin_text = (
        f"{nurse_name} {status} the {offer['shift_type']} shift at "
        f"{offer.get('facility_name') or 'the facility'} on "
        f"{db.pretty_date(offer['date'])}{late_note}."
    )
    try:
        db.send_sms(ADMIN_PHONE, admin_text)
    except Exception:
        logger.exception("Failed to notify admin of adhoc offer reply")

    if answer == "yes":
        return twiml_reply("Great, thanks! Confirmed.")
    return twiml_reply("No worries, thanks for letting us know.")


@app.route("/sms", methods=["POST"])
def sms():
    from_number = request.form.get("From")
    body = request.form.get("Body", "")
    logger.info("SMS from %s: %s", from_number, body)

    # Courtesy: a bare thank-you gets a friendly reply, no shift parsing.
    if body.strip().lower().strip("!.") in ("thanks", "thank you", "ta", "cheers", "thankyou"):
        return twiml_reply(random.choice(
            ["No problem.", "My pleasure.", "Easy.", "No worries.", "Anytime.", "All good."]
        ))

    if _is_admin_number(from_number):
        # Ad hoc admin commands take priority over everything else on his number,
        # including a pending approval — they're a distinct, explicit intent.
        cmd = parse_admin_command(body)
        if cmd.get("type") == "availability":
            return handle_availability_query(cmd)
        if cmd.get("type") == "text_nurse":
            return handle_text_nurse_command(cmd)

        # Paul confirming a pending shift approval — checked before facility/offer
        # routing, since his number is also the Collins callback number (and, in
        # dev/test setups, may also match a stand-in nurse record).
        admin_response = handle_admin_reply(body)
        if admin_response is not None:
            return admin_response

    # An ACTIVE cascade offer on this number wins over facility identity — see the
    # module docstring for why this ordering matters.
    active_offer = db.get_active_offer_by_phone(from_number)
    if active_offer:
        offer_response = handle_offer_reply(active_offer, body)
        if offer_response is not None:
            return offer_response

    # Same priority for an active ad hoc single-nurse offer.
    active_adhoc = db.get_active_adhoc_offer_by_phone(from_number)
    if active_adhoc:
        return handle_adhoc_offer_reply(active_adhoc, body)

    facility = db.facility_by_phone(from_number)

    if not facility:
        # Not a facility, no active offer — could still be a nurse replying late.
        late_response = handle_late_nurse_reply(from_number, body)
        if late_response is not None:
            return late_response

    # Dev: only treat the sender as a stand-in facility if explicitly testing the
    # shift flow. Otherwise an unknown number falls through to the afterhours chat.
    if not facility and db.DEV and os.environ.get("KLARRA_DEV_AS_FACILITY") == "1":
        facility = db.first_facility()
        logger.info("[DEV] unknown SMS sender -> stand-in facility %s",
                    facility["slug"] if facility else None)

    if not facility:
        # Unknown number. Afterhours -> have a conversation; daytime -> brief reply.
        if _afterhours_now():
            return handle_afterhours_chat(from_number, body)
        return twiml_reply("Sorry, this number isn't recognised. Please contact Knightingale directly.")

    callback = from_number
    if db.DEV and from_number not in db.dev_testers():
        callback = os.environ.get("KLARRA_DEV_PHONE", from_number)

    parsed = parse_request(body)
    if not parsed:
        return twiml_reply(
            "Sorry, I couldn't read that. Please text the date, shift "
            "(morning/afternoon/night) and role (RN/EN/PCA/DSW)."
        )

    target_facility_id = facility["id"]
    if facility["slug"] == "collins":
        match = db.find_facility_by_name(parsed.get("facility"))
        if not match:
            return twiml_reply(
                "Which site is this shift for? Please include the facility name, "
                "e.g. 'EN morning shift tomorrow at Port Melbourne'."
            )
        target_facility_id = match["id"]

    try:
        req_id = db.create_shift_request(
            facility_id=target_facility_id,
            callback_number=callback,
            date=parsed["date"],
            shift_type=parsed["shift_type"],
            role=parsed["role"],
            source="sms",
            start_time=parsed.get("start_time"),
            end_time=parsed.get("end_time"),
        )
    except Exception as e:
        logger.exception("Failed to log SMS request")
        return twiml_reply("Something went wrong logging your request. Please try calling instead.")

    return twiml_reply(random.choice(ACK_REPLIES))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
