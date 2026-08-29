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
  5. Pool exhausted -> escalate up the role ladder (PCA -> EN -> RN) and repeat;
     all tiers exhausted -> mark unfilled, call the facility back.

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

# Ranking-failure alerting: text the admins when the model can't rank, but at most
# once per cooldown so a sustained outage doesn't flood the phone.
RANK_ALERT_COOLDOWN_SECONDS = 1800  # 30 min
_last_rank_alert_at = 0.0


def _fallback_order(pool: list[dict]) -> list[dict]:
    """Deterministic order to use when the model can't rank: most reliable first,
    then alphabetical for stable ties. Beats raw Supabase row order, which is
    arbitrary — a degraded night should still put sensible people first."""
    return sorted(
        pool,
        key=lambda n: (-(n.get("reliability") or 0), (n.get("first_name") or "").lower()),
    )


def _alert_rank_failure(req: dict, err: Exception) -> None:
    """Tell the admins that ranking degraded, so an unranked night is visible rather
    than silent. Throttled by RANK_ALERT_COOLDOWN_SECONDS."""
    global _last_rank_alert_at
    now = time.time()
    if now - _last_rank_alert_at < RANK_ALERT_COOLDOWN_SECONDS:
        return
    _last_rank_alert_at = now
    body = (
        "Klarra: nurse ranking unavailable.\n"
        f"Reason: {type(err).__name__}\n"
        f"Request: {req.get('id')} ({req.get('facilities', {}).get('name', '?')}, "
        f"{db.pretty_date(req.get('date', ''))} {req.get('shift_type', '?')})\n\n"
        "Falling back to reliability order. Check OpenAI billing/quota."
    )
    for phone in db.dev_testers():
        try:
            db.send_sms(phone, body)
        except Exception:
            logger.exception("Failed to send rank-failure alert to %s", phone)


def rank_pool(pool: list[dict], req: dict) -> tuple[list[dict], str]:
    """Ask the model to order the eligible pool per the decision skill. Returns
    (ranked_pool, reason_for_top_pick).

    On any failure — quota exhausted, API down, unparseable response — falls back to
    reliability order and alerts the admins. The fallback used to be the pool's raw
    order, which meant a silent quota failure produced a 'top-ranked' nurse that was
    really just whichever row Supabase returned first."""
    client = openai_sdk.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = (
        f"{SKILL}\n\n---\nYou are ranking nurses for this shift: "
        f"{req['role']} {req['shift_type']} on {req['date']} at "
        f"{req['facilities']['name']} (complexity: {req['facilities']['complexity']}).\n"
        f"Here is the eligible pool as JSON:\n{json.dumps(pool)}\n\n"
        f'Return ONLY JSON: {{"order":[nurse_id,...best first],'
        f'"reason":"one short sentence on why the top nurse was chosen, referencing their '
        f'specific attributes (shifts, hours, reliability) — do not say Calling or any action word"}}. No other text.'
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
    except Exception as err:
        logger.exception("Ranking failed; falling back to reliability order")
        _alert_rank_failure(req, err)
        fallback = _fallback_order(pool)
        logger.info("Fallback order: %s", [n["first_name"] for n in fallback])
        return fallback, "ranking unavailable — ordered by reliability"


def rotate_top10(ranked: list[dict]) -> list[dict]:
    """Rotate the full ranked pool by the global counter so the same nurse
    isn't always called first."""
    n = len(ranked)
    if n <= 1:
        return ranked
    offset = db.next_rotation() % n
    return ranked[offset:] + ranked[:offset]


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
        "start_time": req.get("start_time"),
        "end_time": req.get("end_time"),
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
    """Fill a shift, escalating up the role ladder if nobody at the requested level
    takes it (PCA -> EN -> RN). Each tier is fully exhausted before the next is tried,
    and stepping up texts the admins — a higher tier usually bills at a higher rate."""
    fac = req["facilities"]
    logger.info("Filling request %s: %s %s %s at %s",
                req["id"], req["role"], req["shift_type"], req["date"], fac["name"])

    ladder = db.escalation_ladder(req["role"])
    tried_any = False
    waited_afterhours = False

    for role in ladder:
        pool = db.get_candidate_pool(fac["slug"], req["date"], req["shift_type"], role)
        if not pool:
            logger.info("No %s candidates for request %s", role, req["id"])
            continue

        if tried_any:
            notify_escalation(req, role)
        logger.info("Request %s now trying %s tier (%d candidates)",
                    req["id"], role, len(pool))

        ranked, reason = rank_pool(pool, req)
        ranked = rotate_top10(ranked)
        tried_any = True

        if is_afterhours():
            filled = await _handle_afterhours(lk, req, ranked, reason,
                                              skip_wait=waited_afterhours)
            waited_afterhours = True
        else:
            filled = await _handle_daytime(lk, req, ranked, reason)

        if filled:
            if db.DEV:
                db.mark_request_done_dev(req["id"])
            return

    logger.info("All tiers exhausted for request %s", req["id"])
    db.mark_request_unfilled(req["id"])
    await notify_facility(lk, req, filled=False, nurse_name=None)
    if db.DEV:
        db.mark_request_done_dev(req["id"])


def notify_escalation(req: dict, new_role: str) -> None:
    """Tell the admins the shift has moved up a rung, so a higher-rate carer is never
    offered a shift without anyone knowing."""
    body = (
        f"Escalating shift — no {req['role']} took it.\n"
        f"Now offering to: {new_role}\n"
        f"Facility: {req['facilities']['name']}\n"
        f"Date: {db.pretty_date(req['date'])}\n"
        f"Shift: {req['shift_type']}"
    )
    for phone in db.dev_testers():
        try:
            db.send_sms(phone, body)
        except Exception:
            logger.exception("Failed to send escalation alert to %s", phone)


async def _handle_afterhours(lk, req, ranked, reason, skip_wait: bool = False) -> bool:
    """Wait 3 min from when the shift was logged, then assign the top available
    candidate (no nurse call — availability = assignment) and call the facility back.

    Returns True if the shift was filled. Returns False if every candidate in this
    tier was taken, so the caller can escalate; the caller owns the unfilled path.
    skip_wait avoids re-serving the 3-minute hold on a second or third tier."""
    if not skip_wait:
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
        logger.info("All candidates in this tier taken for request %s", req["id"])
        return False

    logger.info("Selected %s for request %s (%s)", top["first_name"], req["id"], reason)

    db.mark_request_filled(req["id"], top["nurse_id"])
    await notify_facility(lk, req, filled=True, nurse_name=top["first_name"])
    send_fyi(req, top, reason, ranked=ranked)
    return True


async def _handle_daytime(lk, req, ranked, reason) -> bool:
    """Call ranked nurses one at a time until one accepts. Availability is only
    flipped to 'assigned' on accept, via the conditional write (race-safe).

    Returns True if the shift was filled, False if this tier is exhausted."""
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
                send_fyi(req, candidate, reason, ranked=ranked)
                return True
            logger.info("Nurse %s accepted but slot was already taken, trying next",
                        candidate["first_name"])
        else:
            logger.info("Nurse %s outcome: %s, trying next", candidate["first_name"], outcome)

    logger.info("Tier exhausted for request %s", req["id"])
    return False


def send_fyi(req, nurse, reason, ranked=None):
    """Text Paul an FYI after the facility has been told. No action needed."""
    admin = os.environ.get("KLARRA_DEV_PHONE")
    if not admin:
        return
    start = req.get("start_time") or ""
    end = req.get("end_time") or ""
    shift_time = f"{start} - {end}" if start and end else req["shift_type"]
    body = (
        f"Shift filled\n"
        f"Nurse: {nurse['first_name']}\n"
        f"Facility: {req['facilities']['name']}\n"
        f"Date: {db.pretty_date(req['date'])}\n"
        f"Shift: {shift_time}\n"
    )
    if ranked:
        body += "Nurses:\n"
        for i, n in enumerate(ranked[:10], 1):
            marker = "✓ " if n["nurse_id"] == nurse["nurse_id"] else ""
            body += f"{i} - {marker}{n['first_name']}: {int(n.get('reliability') or 0)}\n"
    try:
        db.send_sms(admin, body.strip())
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
