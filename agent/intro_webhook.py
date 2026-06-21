"""
Intro webhook — Bubble posts a carer's name + phone, Klarra rings them to introduce
herself. Reuses the outbound agent with kind="intro".

POST /intro  JSON: {"name": "Ruby", "phone": "+61..."}

In dev, the call is redirected to KLARRA_DEV_PHONE.

Run:  gunicorn --bind 0.0.0.0:$PORT --chdir agent intro_webhook:app
"""

import os
import json
import time
import asyncio
import logging

import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from livekit import api

import db

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("knightingale-intro")

app = Flask(__name__)

AGENT_NAME = "knightingale-outbound"


def normalise_au(phone: str) -> str:
    """Normalise an Australian number to E.164 (+61...). Leaves already-+ numbers alone."""
    p = "".join(phone.split())  # strip spaces
    if p.startswith("+"):
        return p
    if p.startswith("0"):
        return "+61" + p[1:]
    if p.startswith("61"):
        return "+" + p
    if p.startswith("4") and len(p) == 9:  # bare mobile, no leading 0
        return "+61" + p
    return p


def _dispatch_intro(name: str, phone: str):
    """Dispatch the outbound agent to place an intro call. Runs its own event loop."""
    async def go():
        lk = api.LiveKitAPI()
        room = f"intro-{int(time.time())}"
        meta = {"kind": "intro", "phone": phone, "nurse_name": name}
        await lk.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=AGENT_NAME, room=room, metadata=json.dumps(meta)
            )
        )
        await lk.aclose()
    asyncio.run(go())


@app.route("/intro", methods=["POST"])
def intro():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    phone = (data.get("phone") or "").strip()
    if not phone:
        return jsonify({"error": "phone required"}), 400
    phone = normalise_au(phone)

    # In dev, ring the dev phone instead of the real carer.
    if db.DEV:
        phone = os.environ.get("KLARRA_DEV_PHONE", phone)
        logger.info("[DEV] intro call redirected to %s", phone)

    try:
        _dispatch_intro(name or "there", phone)
    except Exception as e:
        logger.exception("Failed to dispatch intro call")
        return jsonify({"error": str(e)}), 500

    logger.info("Intro call dispatched: %s (%s)", name, phone)
    return jsonify({"ok": True})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
