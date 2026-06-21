"""
Knightingale scheduling agent — INBOUND intake.

This agent answers calls from facilities. It identifies the facility by caller ID,
gathers the shift request (date, shift type, role), writes it to the shift_requests
queue, tells the caller it'll ring back, and ends the call. The actual nurse-calling
and decision-making happens OFF this call, in the orchestrator (separate process).

Run locally:  python agent/main.py dev
"""

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
2. The facility is already identified by their phone number (see THIS CALL below). Don't
   ask which facility they are.
3. Gather the date and the shift type (Morning/Afternoon/Night). For role: Knightingale
   only staffs EN (enrolled nurses) and RN (registered nurses) on these calls. If the
   caller says anything that sounds like "EN", "enrolled", "AIN", "assistant", or you're
   unsure, treat it as EN. Only use RN if they clearly say "registered" or "RN". Never
   record AIN — it is always EN.
4. Once you have date, shift type, and role, call submit_shift_request to log it. The tool
   gives you a callback time in minutes — use that exact number.
5. THEN, before ending, say the full closing in ONE turn: read the details back AND give
   the callback time AND sign off. For example: "Righto — that's an EN for a morning shift
   on the 20th at Port Melbourne. I'll check who's available and call you back in about 7
   minutes. No worries, we're on it." Always include the callback minutes in this closing.
6. ONLY AFTER you have spoken that entire closing line do you call end_call. Never call
   end_call before the callback time has been said out loud. Do not hang up mid-sentence.

Do NOT look up nurses, name nurses, or promise a specific person. You are taking the
order; the calling-around happens after you hang up.
""".strip()


# Set at call start so the submit tool knows the facility + where to call back, without
# the model having to supply (and possibly mishear) those values.
_call_ctx: dict = {"facility_id": None, "callback_number": None, "facility_slug": None}


@function_tool
async def submit_shift_request(date: str, shift_type: str, role: str) -> str:
    """
    Log the shift request to the queue so the team can start finding a nurse. Call this
    once you have the date, shift type, and role confirmed.

    Args:
        date: shift date in YYYY-MM-DD format.
        shift_type: 'Morning', 'Afternoon', or 'Night'.
        role: 'EN', 'RN', or 'AIN'.
    """
    callback = _call_ctx["callback_number"]
    if not callback:
        return "I can't log this because I couldn't read your callback number."
    try:
        req_id = db.create_shift_request(
            facility_id=_call_ctx["facility_id"],
            callback_number=callback,
            date=date,
            shift_type=shift_type,
            role=role,
        )
    except Exception as e:
        logger.exception("Failed to create shift request")
        return f"Something went wrong logging the request: {e}"
    callback_minutes = random.randint(5, 10)
    return (
        f"Logged request #{req_id}. When you read the date back, say it exactly as "
        f"'{db.pretty_date(date)}'. Tell the caller you'll call back in about "
        f"{callback_minutes} minutes, read the shift details back, then call end_call."
    )


@function_tool
async def end_call() -> str:
    """
    Hang up the call. Only call this AFTER you have spoken the full closing line including
    the callback time and sign-off. Ends the call cleanly.
    """
    import asyncio
    try:
        # Give any final spoken words a moment to play out before the line drops.
        await asyncio.sleep(4)
        ctx = get_job_context()
        await ctx.api.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))
    except Exception as e:
        logger.exception("end_call failed")
        return f"Couldn't end the call: {e}"
    return "Call ended."


async def entrypoint(ctx: JobContext):
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

    if db.DEV:
        _call_ctx["callback_number"] = os.environ.get("KLARRA_DEV_PHONE", caller_number)

    if known_facility:
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

    session = AgentSession(
        llm=openai.realtime.RealtimeModel(voice="alloy"),
    )

    await session.start(
        agent=Agent(
            instructions=INSTRUCTIONS + "\n\n--- THIS CALL ---\n" + caller_context,
            tools=[submit_shift_request, end_call],
        ),
        room=ctx.room,
    )

    await session.generate_reply(
        instructions="Say exactly: You've phoned Knightingale, Klarra speaking. Then stop and wait for the caller."
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name="knightingale-inbound"))
