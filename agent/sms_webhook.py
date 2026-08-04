"""
SMS webhook — receives a facility's text, parses the shift request, logs it, replies.
The orchestrator picks it up and (for sms) texts the facility the result.

Also routes two other kinds of inbound SMS for the SMS nurse-offer workflow:
  - A nurse replying YES/NO to an active offer -> updates sms_nurse_offers.
  - Paul replying OK to an admin-approval request -> updates sms_shift_state.
Both are checked BEFORE the facility-shift-request parsing, since Paul's own number
doubles as the Collins head-office callback number — otherwise an "OK" reply from him
would fall through and get mis-parsed as a shift request.

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
    """Use Claude to pull date/shift/role/times/facility from the text. Returns dict or None."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": (
                "Extract a nursing shift request from this SMS. Knightingale only staffs "
                "EN and RN; treat anything like AIN/assistant as EN. Respond ONLY with JSON: "
                '{"date":"YYYY-MM-DD","shift_type":"Morning|Afternoon|Night","role":"EN|RN",'
                '"start_time":"HH:MM or null","end_time":"HH:MM or null",'
                '"facility":"site name as written in the SMS, or null if none is mentioned"}. '
                "start_time/end_time are 24-hour HH:MM if the SMS states or clearly implies "
                "specific times (e.g. '2-10pm', 'from 14:00'), otherwise null — do not guess. "
                "If you cannot determine date, shift_type, and role, respond {\"error\":\"...\"}. "
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
    return data


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


def handle_nurse_offer_reply(phone: str, body: str) -> Response | None:
    """If this nurse has an active (offered/alerted) SMS offer, route YES/NO to it.
    If they're a known nurse but their offer window has already closed (e.g. a reply
    that arrives after the 40s offer window, or after the shift moved to the next
    candidate), give a clear reply instead of falling through to the generic
    unrecognised-number message. Returns None only if this phone has never received
    an SMS offer at all (caller should fall through)."""
    offer = db.get_active_offer_by_phone(phone)
    if offer:
        answer = _parse_yes_no(body)
        if answer == "yes":
            db.mark_offer(offer["id"], "accepted", replied_at="now()")
            return twiml_reply("Great, thanks! You'll see it in the app shortly.")
        if answer == "no":
            db.mark_offer(offer["id"], "declined", replied_at="now()")
            return twiml_reply("No worries, thanks for letting us know.")
        return twiml_reply("Sorry, I didn't catch that — reply YES or NO for the shift.")

    latest = db.get_latest_offer_by_phone(phone)
    if latest:
        return twiml_reply(
            "Thanks for the reply — that shift's already been sorted, so there's "
            "nothing to action there. We'll text you next time something suitable "
            "comes up."
        )

    return None


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

    # Paul confirming a pending shift approval — checked before facility parsing,
    # since his number is also the Collins callback number.
    if _is_admin_number(from_number):
        admin_response = handle_admin_reply(body)
        if admin_response is not None:
            return admin_response

    # A nurse replying to an active SMS shift offer.
    nurse_response = handle_nurse_offer_reply(from_number, body)
    if nurse_response is not None:
        return nurse_response

    facility = db.facility_by_phone(from_number)

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
        return twiml_reply("Sorry, I couldn't read that. Please text the date, shift (morning/afternoon/night) and role (EN/RN).")

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
