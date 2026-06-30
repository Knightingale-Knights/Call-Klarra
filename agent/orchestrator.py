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
    """Ask the model to order the eligible pool per the decision skill. Returns
    (ranked_pool, reason_for_top_pick). Falls back to original order on any error."""
    client = openai_sdk.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = (
        f"{SKILL}\n\n---\nYou are ranking nurses for this shift: "
        f"{req['role']} {req['shift_type']} on {req['date']} at "
        f"{req['facilities']['name']} (complexity: {req['facilities']['complexity']}).\n"
        f"Here is the eligible pool as JSON:\n{json.dumps(pool)}\n\n"
        f'Return ONLY JSON: {{"order":[nurse_id,...best first],'
        f'"reason":"one short sentence on why the top nurse was chosen, citing the '
        f'specific factors from the policy"}}. No other text.'
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        order = parsed.get("order", [])
        reason = parsed.get("reason", "")
        by_id = {n["nurse_id"]: n for n in pool}
        ranked = [by_id[i] for i in order if i in by_id]
        for n in pool:
            if n not in ranked:
                ranked.append(n)
        logger.info("Ranked order: %s", [n["first_name"] for n in ranked])
        return ranked, reason
    except Exception:
        logger.exception("Ranking failed; using pool order")
        return pool, ""


def rotate_top10(ranked: list[dict]) -> list[dict]:
    """Rotate the top 10 of the ranked pool by the global counter, so the same
    top pick doesn't get called every time. Positions 11+ untouched."""
    n = min(10, len(ranked))
    if n <= 1:
        return ranked
    offset = db.next_rotation() % n
    top = ranked[:n]
    rotated = top[offset:] + top[:offset]
    return rotated + ranked[n:]


async def dispatch_nurse_call(lk: api.LiveKitAPI, nurse: dict, req: dict) -> None:
    """Dispatch the outbound agent + place the call. Does NOT wait for an outcome."""
    room = f"nurse-call-{req['id']}-{nurse['nurse_id']}-{int(time.time())}"
    meta = {
        "kind": "nurse",
        "phone": nurse["phone"],
        "nurse_id": nurse["nurse_id"],
        "nurse_name": nurse["first_name"],
        "request_id": req["id"],
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


AFTERHOURS_START = 21  # 9pm
AFTERHOURS_END = 5     # 5am
CALLBACK_DELAY_SECONDS = 180  # 3 minutes from when the shift was logged


def is_afterhours() -> bool:
    """True if within the afterhours window (9pm–5am Melbourne), forced for testing,
    or running in dev (so the afterhours flow can always be tested)."""
    if db.DEV or os.environ.get("KLARRA_FORCE_AFTERHOURS") == "1":
        return True
    from datetime import datetime
    from zoneinfo import ZoneInfo
    h = datetime.now(ZoneInfo("Australia/Melbourne")).hour
    return h >= AFTERHOURS_START or h < AFTERHOURS_END


def _logged_at(req: dict) -> float:
    """Epoch seconds when the request was created, for the 3-min timer."""
    from datetime import datetime
    raw = req.get("created_at")
    if not raw:
        return time.time()
    try:
        s = str(raw).replace("Z", "+00:00")
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return time.time()


async def handle_request(lk: api.LiveKitAPI, req: dict):
    fac = req["facilities"]
    logger.info("Filling request %s: %s %s %s at %s",
                req["id"], req["role"], req["shift_type"], req["date"], fac["name"])

    pool = db.get_candidate_pool(fac["slug"], req["date"], req["shift_type"], req["role"])
    if not pool:
        logger.info("No eligible nurses for request %s", req["id"])
        db.mark_request_unfilled(req["id"])
        await notify_facility(lk, req, filled=False, nurse_name=None)
        if db.DEV:
            db.mark_request_done_dev(req["id"])
        return

    ranked, reason = rank_pool(pool, req)
    ranked = rotate_top10(ranked)

    if is_afterhours():
        await _handle_afterhours(lk, req, ranked, reason)
    else:
        await _handle_daytime(lk, req, ranked, reason)


async def _handle_afterhours(lk, req, ranked, reason):
    """Wait 3 min from when the shift was logged, then assign the top available
    candidate (no nurse call — availability = assignment) and call the facility back."""
    elapsed = time.time() - _logged_at(req)
    remaining = CALLBACK_DELAY_SECONDS - elapsed
    if remaining > 0:
        logger.info("Afterhours: holding %ds before facility callback", int(remaining))
        await asyncio.sleep(remaining)

    # Try each ranked nurse in order; skip any that got taken in the meantime
    # (conditional update only succeeds if their availability is still 'pending').
    top = None
    for candidate in ranked:
        if db.assign_availability(candidate["nurse_id"], req["date"], req["shift_type"]):
            top = candidate
            break
        logger.info("Nurse %s no longer available for request %s, trying next",
                    candidate["first_name"], req["id"])

    if not top:
        logger.info("All candidates taken for request %s", req["id"])
        db.mark_request_unfilled(req["id"])
        await notify_facility(lk, req, filled=False, nurse_name=None)
        if db.DEV:
            db.mark_request_done_dev(req["id"])
        return

    logger.info("Selected %s for request %s (%s)", top["first_name"], req["id"], reason)

    db.mark_request_filled(req["id"], top["nurse_id"])
    await notify_facility(lk, req, filled=True, nurse_name=top["first_name"])
    send_fyi(req, top, reason)
    if db.DEV:
        db.mark_request_done_dev(req["id"])


async def _handle_daytime(lk, req, ranked, reason):
    """Call ranked nurses one at a time until one accepts. Availability is only
    flipped to 'assigned' on accept, via the conditional write (race-safe)."""
    for candidate in ranked:
        logger.info("Calling %s for request %s", candidate["first_name"], req["id"])
        outcome = await call_one_nurse(lk, candidate, req)
        db.record_call_event(candidate["nurse_id"], outcome,
                              facility_id=req["facility_id"], shift_date=req["date"])
        if outcome == "accepted":
            if db.assign_availability(candidate["nurse_id"], req["date"], req["shift_type"]):
                logger.info("Selected %s for request %s (%s)",
                            candidate["first_name"], req["id"], reason)
                db.mark_request_filled(req["id"], candidate["nurse_id"])
                await notify_facility(lk, req, filled=True, nurse_name=candidate["first_name"])
                send_fyi(req, candidate, f"top-ranked after rotation (original reason: {reason})")
                if db.DEV:
                    db.mark_request_done_dev(req["id"])
                return
            logger.info("Nurse %s accepted but slot was already taken, trying next",
                        candidate["first_name"])
        else:
            logger.info("Nurse %s outcome: %s, trying next", candidate["first_name"], outcome)

    logger.info("All candidates exhausted for request %s", req["id"])
    db.mark_request_unfilled(req["id"])
    await notify_facility(lk, req, filled=False, nurse_name=None)
    if db.DEV:
        db.mark_request_done_dev(req["id"])


def send_fyi(req, nurse, reason):
    """Text Paul an FYI after the facility has been told. No action needed."""
    admin = os.environ.get("KLARRA_DEV_PHONE")
    if not admin:
        return
    body = (
        f"Shift filled.\n"
        f"Facility: {req['facilities']['name']}\n"
        f"Date: {db.pretty_date(req['date'])}\n"
        f"Shift: {req['shift_type']}\n"
        f"Nurse: {nurse['first_name']}\n"
        f"Why: {reason or 'top-ranked per policy'}"
    )
    try:
        db.send_sms(admin, body)
    except Exception:
        logger.exception("Failed to send FYI SMS")


def send_approval_brief(req, nurse, reason):
    """Text Paul the brief and ask for YES/NO before the facility is told."""
    admin = os.environ.get("KLARRA_DEV_PHONE")
    if not admin:
        logger.warning("No KLARRA_DEV_PHONE set; cannot send approval brief")
        return
    body = (
        f"Shift approval needed.\n"
        f"Facility: {req['facilities']['name']}\n"
        f"Date: {db.pretty_date(req['date'])}\n"
        f"Shift: {req['shift_type']}\n"
        f"Nurse: {nurse['first_name']}\n"
        f"Why: {reason or 'top-ranked per policy'}\n\n"
        f"Reply YES to confirm, NO to cancel."
    )
    db.send_sms(admin, body)


async def notify_facility(lk, req, filled, nurse_name):
    """Tell the facility the result — by SMS if the request came via SMS, else voice."""
    if req.get("source") == "sms":
        if filled:
            body = (f"Good news — {nurse_name} is covering the {req['shift_type'].lower()} "
                    f"shift on {db.pretty_date(req['date'])}.")
        else:
            body = (f"Sorry, no one was available for the {req['shift_type'].lower()} shift "
                    f"on {db.pretty_date(req['date'])} yet. We'll keep trying.")
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
