"""
Outbound agent — handles outbound calls for Knightingale (explicit dispatch).

ONE worker, two job types selected by dispatch metadata:
  - "nurse"    : call a nurse, offer the shift, record accepted/declined.
  - "facility" : call the facility back, tell them the result.

The orchestrator (orchestrator.py) dispatches this agent into a room, then places the
SIP call into that same room. This agent greets, talks, records the outcome into a
Supabase row the orchestrator reads, and hangs up.

Outcome handoff: for nurse calls, the agent writes the outcome onto the shift_requests
row's transient store via call_events + a status flag the orchestrator polls. To keep it
simple and robust, the nurse agent writes the outcome into a dedicated 'call_outcomes'
mechanism: we reuse call_events (authoritative) AND set room metadata is not persistent,
so the orchestrator polls call_events for this nurse+request.

Run:  python agent/outbound.py dev
"""

import os
import json
import asyncio
import logging

import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from dotenv import load_dotenv
from livekit import api
from livekit.agents import (
    Agent, AgentSession, JobContext, WorkerOptions, cli, function_tool, get_job_context,
)
from livekit.plugins import openai
try:
    from openai.types.beta.realtime.session import TurnDetection
except Exception:
    TurnDetection = None

import db

load_dotenv()
logger = logging.getLogger("knightingale-agent.outbound")
logger.setLevel(logging.INFO)
for noisy in ("hpack", "httpx", "httpcore", "h2"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

AGENT_NAME = "knightingale-outbound"


async def _hang_up():
    await asyncio.sleep(6)
    jc = get_job_context()
    await jc.api.room.delete_room(api.DeleteRoomRequest(room=jc.room.name))


def nurse_instructions(d: dict) -> str:
    return f"""
You are Klarra from Knightingale, calling a nurse to offer a shift. Warm, brief. Speak with an English (British) accent.
Speak naturally and let your sentences finish.
1. Open with the offer in one line: "Hi {d['nurse_name']}, I've got a {d['role']} {d['shift_type'].lower()} shift on {d['date']} at {d['facility_name']} — would you like it?"
2. Wait for their answer.
3. If YES: FIRST say your full closing out loud, e.g. "Great, thanks {d['nurse_name']} — you'll see the shift pop up in the app shortly. Have a good one!" THEN call record_result with "accepted". THEN call hang_up.
4. If NO: FIRST say a full friendly sign-off out loud, e.g. "No worries at all, thanks {d['nurse_name']}, have a good one!" THEN call record_result with "declined". THEN call hang_up.
Always speak your full line before any tool call. Never cut yourself off. Never hang up mid-sentence.
""".strip()


def facility_instructions(d: dict) -> str:
    if d.get("filled"):
        body = (f"Good news — you've got {d['nurse_name']} covering the "
                f"{d['shift_type'].lower()} shift on {d['date']}.")
    else:
        body = (f"Unfortunately I couldn't find anyone available for the "
                f"{d['shift_type'].lower()} shift on {d['date']} just yet. "
                f"I'll keep trying and let you know.")
    return f"""
You are Klarra from Knightingale, calling a facility back with an update. Warm, brief.
1. Open: "Hi, it's Klarra from Knightingale calling you back."
2. Tell them: {body}
3. Ask if there's anything else. After they respond, say a full sign-off out loud, e.g.
   "No worries, we're on it — have a good one!" ONLY after speaking that whole line do you
   call hang_up. Never hang up before your closing sentence is fully spoken.
""".strip()


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    meta = json.loads(ctx.job.metadata or "{}")
    kind = meta.get("kind")
    phone = meta.get("phone")
    logger.info("Outbound job: kind=%s phone=%s room=%s", kind, phone, ctx.room.name)

    # Place the outbound call into this room.
    trunk_id = os.environ["OUTBOUND_TRUNK_ID"]
    try:
        await ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=trunk_id,
                sip_call_to=phone,
                participant_identity="callee",
                wait_until_answered=True,
            )
        )
    except api.TwirpError:
        logger.warning("Call to %s not answered/failed", phone)
        if kind == "nurse":
            db.record_call_event(meta["nurse_id"], "no_answer",
                                 facility_id=meta.get("facility_id"),
                                 shift_date=meta.get("date"))
        await ctx.shutdown()
        return

    result_store = {"value": None}

    @function_tool
    async def record_result(outcome: str) -> str:
        """Record the nurse's answer. Args: outcome: 'accepted' or 'declined'."""
        result_store["value"] = outcome
        if db.DEV:
            if meta.get("request_id"):
                db.set_dev_outcome(meta["request_id"], outcome)
        else:
            db.record_call_event(meta["nurse_id"], outcome,
                                 facility_id=meta.get("facility_id"),
                                 shift_date=meta.get("date"))
        return "Recorded."

    @function_tool
    async def hang_up() -> str:
        """End the call after you've finished speaking."""
        await _hang_up()
        return "Ended."

    tools = [hang_up]
    if kind == "nurse":
        tools = [record_result, hang_up]
        instructions = nurse_instructions(meta)
        greet = f"Greet {meta['nurse_name']} and offer the shift."
    else:
        instructions = facility_instructions(meta)
        greet = "Greet the facility and give them the update."

    if TurnDetection is not None:
        rt = openai.realtime.RealtimeModel(
            voice="alloy",
            speed=1.25,
            turn_detection=TurnDetection(
                type="semantic_vad",
                eagerness="low",
                create_response=True,
                interrupt_response=True,
            ),
        )
    else:
        rt = openai.realtime.RealtimeModel(voice="alloy", speed=1.25)
    session = AgentSession(llm=rt)
    await session.start(agent=Agent(instructions=instructions, tools=tools), room=ctx.room)
    await session.generate_reply(instructions=greet)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name=AGENT_NAME))
