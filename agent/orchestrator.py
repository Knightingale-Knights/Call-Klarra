"""
Orchestrator — the brain that fills shift requests off-call.

Loop:
  1. Claim the oldest pending shift_request (marks it 'working').
  2. Get the eligible pool (get_candidate_pool — hard filters already applied).
  3. Ask the model to RANK the pool per the decision skill (explainable).
  4. For each nurse in order: dispatch the outbound agent + place the call, wait for the
     accepted/declined/no_answer outcome (read from call_events).
     - accepted -> mark filled, call the facility back with the good news, stop.
     - else     -> next nurse.
  5. Pool exhausted -> mark unfilled, call the facility back.

Run:  python agent/orchestrator.py
(Needs the outbound worker running too: python agent/outbound.py dev)
"""

import os
import json
import time
import asyncio
import logging
from pathlib import Path

import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from dotenv import load_dotenv
from livekit import api
import openai as openai_sdk

import db

load_dotenv()
logger = logging.getLogger("knightingale-orchestrator")
logging.basicConfig(level=logging.INFO)

AGENT_NAME = "knightingale-outbound"
SKILL = (Path(__file__).parent.parent / "skills" / "nurse-selection.md").read_text()
POLL_SECONDS = 5


def rank_pool(pool: list[dict], req: dict) -> list[dict]:
    """Ask the model to order the eligible pool per the decision skill. Returns the
    pool re-ordered best-first. Falls back to original order on any error."""
    client = openai_sdk.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = (
        f"{SKILL}\n\n---\nYou are ranking nurses for this shift: "
        f"{req['role']} {req['shift_type']} on {req['date']} at "
        f"{req['facilities']['name']} (complexity: {req['facilities']['complexity']}).\n"
        f"Here is the eligible pool as JSON:\n{json.dumps(pool)}\n\n"
        f"Return ONLY a JSON array of nurse_id values, best first, no other text."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        order = json.loads(text)
        by_id = {n["nurse_id"]: n for n in pool}
        ranked = [by_id[i] for i in order if i in by_id]
        # append any the model missed
        for n in pool:
            if n not in ranked:
                ranked.append(n)
        logger.info("Ranked order: %s", [n["first_name"] for n in ranked])
        return ranked
    except Exception:
        logger.exception("Ranking failed; using pool order")
        return pool


async def dispatch_nurse_call(lk: api.LiveKitAPI, nurse: dict, req: dict) -> None:
    """Dispatch the outbound agent + place the call. Does NOT wait for an outcome."""
    room = f"nurse-call-{req['id']}-{nurse['nurse_id']}-{int(time.time())}"
    meta = {
        "kind": "nurse",
        "phone": nurse["phone"],
        "nurse_id": nurse["nurse_id"],
        "nurse_name": nurse["first_name"],
        "facility_id": req["facility_id"],
        "facility_name": req["facilities"]["name"],
        "date": req["date"],
        "shift_type": req["shift_type"],
        "role": req["role"],
    }
    await lk.agent_dispatch.create_dispatch(
        api.CreateAgentDispatchRequest(
            agent_name=AGENT_NAME, room=room, metadata=json.dumps(meta)
        )
    )


async def call_one_nurse(lk: api.LiveKitAPI, nurse: dict, req: dict) -> str:
    """Dispatch the outbound agent + place the call. Returns the outcome string."""
    await dispatch_nurse_call(lk, nurse, req)
    # Wait for the call to resolve by polling call_events for this nurse.
    deadline = time.time() + 90
    last_seen = _latest_outcome(nurse["nurse_id"])
    while time.time() < deadline:
        await asyncio.sleep(3)
        latest = _latest_outcome(nurse["nurse_id"])
        if latest and latest != last_seen:
            return latest["outcome"]
    return "no_answer"


def _latest_outcome(nurse_id: int):
    client = db.get_client()
    resp = (
        client.table("call_events")
        .select("outcome, occurred_at")
        .eq("nurse_id", nurse_id)
        .order("occurred_at", desc=True)
        .limit(1)
        .execute()
    )
    return resp.data[0] if resp.data else None


async def call_facility(lk: api.LiveKitAPI, req: dict, filled: bool, nurse_name: str | None):
    room = f"facility-cb-{req['id']}-{int(time.time())}"
    meta = {
        "kind": "facility",
        "phone": req["facility_callback_number"],
        "filled": filled,
        "nurse_name": nurse_name or "",
        "date": req["date"],
        "shift_type": req["shift_type"],
    }
    await lk.agent_dispatch.create_dispatch(
        api.CreateAgentDispatchRequest(
            agent_name=AGENT_NAME, room=room, metadata=json.dumps(meta)
        )
    )


async def handle_request(lk: api.LiveKitAPI, req: dict):
    fac = req["facilities"]
    logger.info("Filling request %s: %s %s %s at %s",
                req["id"], req["role"], req["shift_type"], req["date"], fac["name"])

    # DEV: skip pool + ranking. Place ONE call to the dev phone, don't poll for an
    # outcome (call_events writes are blocked in dev), then mark the request done so
    # it can't be re-claimed and call you again.
    if db.DEV:
        dev_phone = os.environ.get("KLARRA_DEV_PHONE")
        if not dev_phone:
            logger.warning("[DEV] no KLARRA_DEV_PHONE set; cannot place test call")
            db.mark_request_done_dev(req["id"])
            return
        nurse = {"nurse_id": -1, "first_name": "there", "phone": dev_phone}
        logger.info("[DEV] calling %s as test nurse (one-shot)", dev_phone)
        await dispatch_nurse_call(lk, nurse, req)
        db.mark_request_done_dev(req["id"])
        logger.info("[DEV] request %s marked done; will not re-call", req["id"])
        return

    pool = db.get_candidate_pool(fac["slug"], req["date"], req["shift_type"], req["role"])
    if not pool:
        logger.info("No eligible nurses for request %s", req["id"])
        db.mark_request_unfilled(req["id"])
        await call_facility(lk, req, filled=False, nurse_name=None)
        return

    for nurse in rank_pool(pool, req):
        logger.info("Calling %s for request %s", nurse["first_name"], req["id"])
        outcome = await call_one_nurse(lk, nurse, req)
        logger.info("%s -> %s", nurse["first_name"], outcome)
        if outcome == "accepted":
            db.mark_request_filled(req["id"], nurse["nurse_id"])
            await notify_facility(lk, req, filled=True, nurse_name=nurse["first_name"])
            return

    db.mark_request_unfilled(req["id"])
    await notify_facility(lk, req, filled=False, nurse_name=None)


async def notify_facility(lk, req, filled, nurse_name):
    """Tell the facility the result — by SMS if the request came via SMS, else voice."""
    if req.get("source") == "sms":
        if filled:
            body = (f"Good news — {nurse_name} is covering the {req['shift_type'].lower()} "
                    f"shift on {req['date']}.")
        else:
            body = (f"Sorry, no one was available for the {req['shift_type'].lower()} shift "
                    f"on {req['date']} yet. We'll keep trying.")
        try:
            db.send_sms(req["facility_callback_number"], body)
        except Exception:
            logger.exception("Failed to send result SMS")
    else:
        await call_facility(lk, req, filled=filled, nurse_name=nurse_name)


async def main():
    lk = api.LiveKitAPI()
    logger.info("Orchestrator running. Polling for requests every %ss.", POLL_SECONDS)
    while True:
        try:
            req = db.claim_next_request()
            if req:
                await handle_request(lk, req)
            else:
                await asyncio.sleep(POLL_SECONDS)
        except Exception:
            logger.exception("Error handling request")
            await asyncio.sleep(POLL_SECONDS)


if __name__ == "__main__":
    asyncio.run(main())
