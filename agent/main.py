"""
Knightingale scheduling agent — INBOUND intake.

This agent answers calls from facilities. It identifies the facility by caller ID,
gathers the shift request (date, shift type, role), writes it to the shift_requests
queue, tells the caller it'll ring back, and ends the call. The actual nurse-calling
and decision-making happens OFF this call, in the orchestrator (separate process).

Run locally:  python agent/main.py dev
"""

import asyncio
import os
import random
import logging
from pathlib import Path

import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from dotenv import load_dotenv

from livekit import api
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli, function_tool, get_job_context
from livekit.plugins import openai

import db

load_dotenv()

logger = logging.getLogger("knightingale-agent")
logger.setLevel(logging.INFO)

for noisy in ("hpack", "httpx", "httpcore", "h2", "hpack.hpack", "hpack.table"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


INSTRUCTIONS = """
You are Klarra, the Knightingale scheduling assistant — a warm, efficient Australian
voice that takes shift requests from aged-care and nursing facilities by phone. Speak
naturally and keep replies short, like a real phone call.

Your ONLY job on this call is to take the request, not to fill it. Specifically:
1. Open with exactly: "You've phoned Knightingale, Klarra speaking." Then STOP and wait
   for the caller to tell you what they need. Do not ask a follow-up question in the same
   breath as the greeting.
2. The facility is already identified by their phone number, UNLESS the THIS CALL section
   below says otherwise (e.g. head office calling on behalf of another site) — follow
   THIS CALL's instructions on this point, they take priority over this general rule.
3. Gather the date and the shift type (Morning/Afternoon/Night). For role: Knightingale
   only staffs EN (enrolled nurses) and RN (registered nurses) on these calls. If the
   caller says anything that sounds like "EN", "enrolled", "AIN", "assistant", or you're
   unsure, treat it as EN. Only use RN if they clearly say "registered" or "RN". Never
   record AIN — it is always EN.
4. Once you have date, shift type, and role, call submit_shift_request to log it. The tool
   gives you a callback time in minutes — use that exact number.
5. THEN say the full closing in ONE turn: read the details back AND give the callback time AND say "Bye!" Then call hang_up. For example: "Righto — that's an EN for a morning shift on Friday, June 26 at Port Melbourne. I'll check who's available and call you back in about 7 minutes. Bye!" Then call hang_up.
6. Never end the call before saying the full closing and calling hang_up.

Do NOT look up nurses, name nurses, or promise a specific person. You are taking the order;
the calling-around happens afterwards.
""".strip()


# Set at call start so the submit tool knows the facility + where to call back, without
# the model having to supply (and possibly mishear) those values.
_call_ctx: dict = {"facility_id": None, "callback_number": None, "facility_slug": None,
                   "is_head_office": False, "facility_lookup": {}}


@function_tool
async def submit_shift_request(date: str, shift_type: str, role: str,
                               start_time: str | None = None,
                               end_time: str | None = None,
                               target_facility_slug: str | None = None) -> str:
    """
    Log the shift request. Call once you have date, shift type, role, and ideally
    start/end times confirmed.

    Args:
        date: YYYY-MM-DD.
        shift_type: 'Morning', 'Afternoon', or 'Night'.
        role: 'EN' or 'RN'.
        start_time: e.g. '14:00' or '2pm'. Optional but ask for it.
        end_time: e.g. '21:30' or '930pm'. Optional but ask for it.
        target_facility_slug: Collins calls only — site name as spoken by caller.
    """
    callback = _call_ctx["callback_number"]
    if not callback:
        return "I can't log this because I couldn't read your callback number."

    facility_id = _call_ctx["facility_id"]
    if _call_ctx["is_head_office"] and target_facility_slug:
        match = _call_ctx["facility_lookup"].get(target_facility_slug.lower().strip())
        if not match:
            return (f"'{target_facility_slug}' isn't a facility I recognise — ask the "
                    f"caller to confirm the site name and try again.")
        facility_id = match["id"]

    try:
        req_id = db.create_shift_request(
            facility_id=facility_id,
            callback_number=callback,
            date=date,
            shift_type=shift_type,
            role=role,
            start_time=start_time,
            end_time=end_time,
        )
    except Exception as e:
        logger.exception("Failed to create shift request")
        return f"Something went wrong logging the request: {e}"
    callback_minutes = random.randint(5, 10)
    return (
        f"Logged request #{req_id}. When you read the date back, say it exactly as "
        f"'{db.pretty_date(date)}'. Tell the caller you'll call back in about "
        f"{callback_minutes} minutes, read the shift details back, then invite them to hang up."
    )


@function_tool
async def hang_up() -> str:
    """End the call 2 seconds after Klarra says Bye."""
    await asyncio.sleep(2)
    jc = get_job_context()
    await jc.api.room.delete_room(api.DeleteRoomRequest(room=jc.room.name))
    return "Ended."
    await ctx.connect()
    logger.info("Agent connected to room: %s", ctx.room.name)

    caller_number = None
    known_facility = None
    try:
        participant = await ctx.wait_for_participant()
        ident = participant.identity or ""
        if ident.startswith("sip_"):
            caller_number = ident.removeprefix("sip_")
        if caller_number:
            known_facility = db.facility_by_phone(caller_number)
            logger.info("Caller %s -> facility %s", caller_number,
                        known_facility["slug"] if known_facility else "UNKNOWN")
    except Exception:
        logger.exception("Caller identification failed")

    # DEV: if caller isn't a known facility, use a stand-in facility so the test
    # flow runs (orchestrator needs a facility row). Real callback goes to dev phone.
    if db.DEV and not known_facility:
        known_facility = db.first_facility()
        if known_facility:
            logger.info("[DEV] unknown caller -> using stand-in facility %s",
                        known_facility["slug"])

    _call_ctx["callback_number"] = caller_number
    _call_ctx["facility_id"] = known_facility["id"] if known_facility else None
    _call_ctx["facility_slug"] = known_facility["slug"] if known_facility else None
    _call_ctx["is_head_office"] = bool(known_facility and known_facility["slug"] == "collins")

    if db.DEV and caller_number not in db.dev_testers():
        _call_ctx["callback_number"] = os.environ.get("KLARRA_DEV_PHONE", caller_number)

    if known_facility and _call_ctx["is_head_office"]:
        all_facilities = [f for f in db.list_facilities() if f["slug"] != "collins"]
        _call_ctx["facility_lookup"] = {}
        for f in all_facilities:
            _call_ctx["facility_lookup"][f["slug"].lower()] = f
            _call_ctx["facility_lookup"][f["name"].lower()] = f
        site_list = ", ".join(f"{f['name']} ({f['slug']})" for f in all_facilities)
        caller_context = (
            f"This call is from Collins, Knightingale's head office. Collins books shifts "
            f"ON BEHALF OF other sites — it is never itself the shift location. Ask the "
            f"caller which site this shift is for. Valid sites: {site_list}. Match what "
            f"they say to the closest name in that list and pass its slug as "
            f"target_facility_slug when you call submit_shift_request. If you're not sure "
            f"which site they mean, ask them to confirm before logging it."
        )
    elif known_facility:
        caller_context = (
            f"This call is from {known_facility['name']} "
            f"(facility slug: {known_facility['slug']}, "
            f"complexity: {known_facility['complexity']}). Greet them by name. The shift "
            f"is for this facility. Do NOT ask which facility they're from."
        )
    else:
        caller_context = (
            f"This caller's number ({caller_number or 'unknown'}) is NOT a recognised "
            f"facility. For security, do not take a shift request. Politely explain you "
            f"can't take the request because the number isn't recognised, and that you'll "
            f"flag it for Paul. Do not call submit_shift_request."
        )

    from livekit.plugins.openai import realtime
    try:
        from openai.types.beta.realtime.session import TurnDetection
        rt = realtime.RealtimeModel(
            voice="shimmer",
            speed=1.25,
            turn_detection=TurnDetection(
                type="semantic_vad",
                eagerness="medium",
                create_response=True,
                interrupt_response=True,
            ),
        )
    except Exception:
        rt = realtime.RealtimeModel(voice="shimmer", speed=1.25)
    session = AgentSession(llm=rt)

    from datetime import datetime
    from zoneinfo import ZoneInfo
    mel = datetime.now(ZoneInfo("Australia/Melbourne"))
    today_line = (
        f"\n\nToday is {mel.strftime('%A, %B ')}{mel.day} ({mel.strftime('%Y-%m-%d')}), "
        f"Melbourne time. Resolve 'today', 'tomorrow', 'this Friday' etc. from that. "
        f"'Tomorrow' is the day after today."
    )

    await session.start(
        agent=Agent(
            instructions=INSTRUCTIONS + today_line + "\n\n--- THIS CALL ---\n" + caller_context,
            tools=[submit_shift_request, hang_up],
        ),
        room=ctx.room,
    )

    await session.generate_reply(
        instructions="Say exactly: You've phoned Knightingale, Klarra speaking. Then stop and wait for the caller."
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name="knightingale-inbound"))
