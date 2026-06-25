"""
SMS webhook — receives a facility's text, parses the shift request, logs it, replies.
The orchestrator picks it up and (for sms) texts the facility the result.

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
    """Use Claude to pull date/shift/role from the text. Returns dict or None."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": (
                "Extract a nursing shift request from this SMS. Knightingale only staffs "
                "EN and RN; treat anything like AIN/assistant as EN. Respond ONLY with JSON: "
                '{"date":"YYYY-MM-DD","shift_type":"Morning|Afternoon|Night","role":"EN|RN"}. '
                "If you cannot determine all three, respond {\"error\":\"...\"}. "
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

    facility = db.facility_by_phone(from_number)
    if not facility and db.DEV:
        facility = db.first_facility()
        logger.info("[DEV] unknown SMS sender -> stand-in facility %s",
                    facility["slug"] if facility else None)
    if not facility:
        return twiml_reply("Sorry, this number isn't recognised. Please contact Knightingale directly.")

    callback = from_number
    if db.DEV:
        callback = os.environ.get("KLARRA_DEV_PHONE", from_number)

    parsed = parse_request(body)
    if not parsed:
        return twiml_reply("Sorry, I couldn't read that. Please text the date, shift (morning/afternoon/night) and role (EN/RN).")

    try:
        req_id = db.create_shift_request(
            facility_id=facility["id"],
            callback_number=callback,
            date=parsed["date"],
            shift_type=parsed["shift_type"],
            role=parsed["role"],
            source="sms",
        )
    except Exception as e:
        logger.exception("Failed to log SMS request")
        return twiml_reply("Something went wrong logging your request. Please try calling instead.")

    return twiml_reply(random.choice(ACK_REPLIES))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
