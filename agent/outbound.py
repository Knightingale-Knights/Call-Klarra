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

AGENT_NAME = os.environ.get("OUTBOUND_AGENT_NAME", "knightingale-outbound")


def text_outcome(meta: dict, outcome: str) -> None:
    """Text Paul each time a nurse call resolves (accepted/declined/no_answer)."""
    admin = os.environ.get("KLARRA_DEV_PHONE")
    if not admin:
        return
    body = (
        f"Nurse call: {outcome}\n"
        f"Nurse: {meta.get('nurse_name', '?')}\n"
        f"Facility: {meta.get('facility_name', '?')}\n"
        f"Date: {db.pretty_date(meta.get('date', ''))}\n"
        f"Shift: {meta.get('shift_type', '?')}"
    )
    try:
        db.send_sms(admin, body)
    except Exception:
        logger.exception("Failed to send outcome SMS")


async def _hang_up():
    await asyncio.sleep(6)
    jc = get_job_context()
    await jc.api.room.delete_room(api.DeleteRoomRequest(room=jc.room.name))


def nurse_instructions(d: dict) -> str:
    nice_date = db.pretty_date(d['date'])
    if d.get("start_time") and d.get("end_time"):
        shift_desc = f"from {d['start_time']} to {d['end_time']}"
    else:
        shift_desc = d['shift_type'].lower()
    return f"""
You are Klarra from Knightingale, calling a nurse to offer a shift. Warm, brief. Speak with an English (British) accent.
Speak naturally and let your sentences finish.
1. Open with the offer in one line: "Hi {d['nurse_name']}, I've got a {d['role']} shift {shift_desc} on {nice_date} at {d['facility_name']} — would you like it?"
2. Wait for their answer. You MUST hear a clear spoken response from the nurse before proceeding. If you hear nothing, silence, a click, or the call drops — call record_result with "declined" immediately. Only proceed to step 3 if the nurse actually speaks and gives a clear affirmative ("yes", "yep", "yeah", "sure", "sounds good", etc.). Any automated message or voicemail = record_result "declined".
3. If YES (clear explicit confirmation from a real person): FIRST say your full closing out loud, e.g. "Great, thanks {d['nurse_name']} — you'll see the shift pop up in the app shortly. Have a good one!" THEN call record_result with "accepted". THEN call hang_up.
4. If NO or unclear: FIRST say a full friendly sign-off out loud, e.g. "No worries at all, thanks {d['nurse_name']}, have a good one!" THEN call record_result with "declined". THEN call hang_up.
Always speak your full line before any tool call. Never cut yourself off. Never hang up mid-sentence.
""".strip()


def facility_instructions(d: dict) -> str:
    nice_date = db.pretty_date(d['date'])
    if d.get("filled"):
        body = (f"Good news — you've got {d['nurse_name']} covering the "
                f"{d['shift_type'].lower()} shift on {nice_date}.")
    else:
        body = (f"Unfortunately I couldn't find anyone available for the "
                f"{d['shift_type'].lower()} shift on {nice_date} just yet. "
                f"I'll keep trying and let you know.")
    return f"""
You are Klarra from Knightingale, calling a facility back with an update. Warm, brief.
1. Open: "Hi, it's Klarra from Knightingale calling you back."
2. Tell them: {body}
3. Ask if there's anything else. After they respond, say a full sign-off and invite them to
   hang up, e.g. "No worries, we're on it. Please feel free to hang up whenever you're ready —
   Paul's asked me never to hang up myself, just in case I cut someone off by accident. Have a
   good one!" Then simply wait. Never end the call yourself.
""".strip()


def intro_instructions(d: dict) -> str:
    name = d.get("nurse_name") or "there"
    return f"""
You are Klarra, Knightingale's new AI scheduling assistant, calling a carer named {name} to introduce yourself. Warm, friendly, brief. Speak with an English (British) accent. This is a first hello, not a shift offer.
1. Open: "Hi {name}, it's Klarra from Knightingale — I'm the new assistant who'll be helping organise shifts. I just wanted to quickly introduce myself."
2. Explain in a sentence or two: from now on, when a shift comes up that suits them, you'll give them a quick call or text to offer it — they can simply say yes or no, and if they say yes it'll appear in their app.
3. Warmly invite them to say yes when they hear from you, and say there's nothing they need to do right now.
4. Ask if they have any quick questions. Answer briefly and naturally if they do.
5. When the conversation is winding up, say a full friendly sign-off and invite them to hang up, e.g. "Lovely — I'll be in touch when something suitable comes up. Please feel free to hang up whenever you're ready; Paul's asked me never to hang up myself in case I cut someone off by accident. Have a good one, {name}!" Then simply wait. Never end the call yourself.
Keep it short and human.
""".strip()


def finalise_questionnaire(meta: dict, state: dict) -> None:
    """Close the session once: Completed if nothing asked is still empty, else Partial."""
    import questionnaire as q
    from datetime import datetime, timezone
    if state.get("closed"):
        return
    state["closed"] = True
    state["finished"] = True
    try:
        sess = q.get_session(meta["session_id"])
        if not sess:
            return
        rec = q.get_record(meta["section"], meta["participant_bubble_id"]) or {}
        still = [k for k in meta["fields"] if k in q.missing_fields(meta["section"], rec)]
        q.update_session(
            sess,
            status="Completed" if not still else "Partial",
            completed=datetime.now(timezone.utc).isoformat(),
            skipped_fields=",".join(sorted(set(state["skipped"]))),
            transcript="\n".join(state["transcript"])[:50000],
        )
    except Exception:
        logger.exception("Failed to finalise questionnaire session %s", meta.get("session_id"))


def build_questionnaire_tools(meta: dict):
    import questionnaire as q
    section = meta["section"]
    allowed = set(meta["fields"]) | {"home safety notes"}
    state = {"transcript": [], "skipped": [], "finished": False, "closed": False}

    @function_tool
    async def save_answer(field: str, value: str) -> str:
        """Save one answer. Args: field: the field name exactly as listed. value: the answer."""
        field = field.strip().lower()
        if field not in allowed:
            return f"Unknown field '{field}'. Use one of: {sorted(allowed)}"
        try:
            fields, msg = q.normalise_answer(section, field, value)
        except ValueError as e:
            return f"Not saved: {e}"
        try:
            await asyncio.to_thread(q.save_fields, section, meta["record_id"],
                                    meta["participant_bubble_id"], fields,
                                    field == "home safety notes")
        except Exception:
            logger.exception("save_answer failed for %s", field)
            return "Could not save that one. Carry on, it will be asked again another time."
        if field in state["skipped"]:
            state["skipped"].remove(field)
        return msg

    @function_tool
    async def skip_question(field: str, reason: str) -> str:
        """Skip a question the carer can't answer. Args: field: field name. reason: short reason."""
        state["skipped"].append(field.strip().lower())
        return "Skipped."

    @function_tool
    async def finish_call() -> str:
        """Call after you have said your full goodbye. Saves the session and ends the call."""
        await asyncio.to_thread(finalise_questionnaire, meta, state)
        await _hang_up()
        return "Ended."

    return [save_answer, skip_question, finish_call], state


QUESTIONNAIRE_MAX_SECONDS = 15 * 60
QUESTIONNAIRE_WRAP_SECONDS = 13 * 60


def questionnaire_instructions(d: dict) -> str:
    import questionnaire as q
    section = d["section"]
    p = d["participant_name"]
    lines = []
    for key in d["fields"]:
        f = q.field_def(section, key)
        if not f:
            continue
        if f["kind"] == "list":
            extra = f" [list; options: {', '.join(q.OPTIONS[f['options']])}]"
        elif f["kind"] == "answer":
            extra = " [Yes / No / Unknown]"
        elif f["kind"] == "safety":
            extra = " [Safe / Unsafe / N/A]"
        else:
            extra = " [free text]"
        sens = " (sensitive)" if f["sensitive"] else ""
        lines.append(f"- field \"{key}\": {f['question']}{extra}{sens}")
    questions = "\n".join(lines)
    home = section == "Home Safety"
    who = ("These are about the home itself, so the carer can answer from what they can see. "
           if home else
           f"The carer is with {p} right now. Ask the carer to ask {p} each question, or answer "
           f"from what they already know, then tell you. ")
    return f"""
You are Klarra from Knightingale, calling {d['carer_name']}, a carer who is on shift with {p}. Speak with an English (British) accent. Warm, clear and efficient. You texted {d['carer_name']} yesterday to say you'd call now.
Purpose: fill in missing {section} information for {p}. {who}

Flow:
1. Open: "Hi {d['carer_name']}, it's Klarra from Knightingale, as promised. Is now still OK for a few quick questions about {p}? It'll take no more than about 15 minutes."
   If now is not OK, say a friendly goodbye and call finish_call.
2. Ask the questions below, one at a time, in order. Keep each question short and natural.
3. After each answer, call save_answer with the field name exactly as written below and the answer.
   - Free text: a short clear summary of what they said. Use "None" if there is nothing (e.g. no allergies).
   - Yes / No / Unknown and Safe / Unsafe / N/A: pass exactly one of those words.
   - List fields: ask openly first; if they're unsure, mention 3 or 4 common options. Pass a comma separated list using the option names where they fit, and their own words for anything else. Pass "None" if none apply.
   - If something is Unsafe, ask briefly what the problem is, then also call save_answer for "home safety notes" with a one line note (e.g. "Stairs: loose railing on back steps").
   - If they don't know, call skip_question with the field and a short reason, and move on. Never guess or invent an answer.
4. Sensitive topics (marked sensitive): tell the carer they can answer from their own knowledge, or step away first, rather than asking out loud if that would be uncomfortable.
5. Never give medical advice. If they raise a safety concern or incident, thank them, note it in the closest free text answer, and suggest they also record it in a progress note.
6. When all questions are done, or you're told time is up: thank them, say anything left will be picked up on another shift, give a full warm goodbye out loud, THEN call finish_call.
Always say your full line before any tool call. Never cut yourself off.

Questions:
{questions}
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
        if kind == "questionnaire":
            import questionnaire as q
            sess = q.get_session(meta["session_id"])
            if sess:
                q.update_session(sess, status="No Answer")
            return
        if kind == "nurse":
            db.record_call_event(meta["nurse_id"], "no_answer",
                                 facility_id=meta.get("facility_id"),
                                 shift_date=meta.get("date"))
            text_outcome(meta, "no_answer")
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
            text_outcome(meta, outcome)
        return "Recorded."

    def on_participant_disconnected(participant):
        """If nurse disconnects before record_result fired, log no_answer."""
        if participant.identity == "callee" and result_store["value"] is None:
            logger.info("Callee disconnected before result — logging no_answer")
            if kind == "nurse" and not db.DEV:
                db.record_call_event(meta["nurse_id"], "no_answer",
                                     facility_id=meta.get("facility_id"),
                                     shift_date=meta.get("date"))
                text_outcome(meta, "no_answer")
            result_store["value"] = "no_answer"

    ctx.room.on("participant_disconnected", on_participant_disconnected)

    @function_tool
    async def hang_up() -> str:
        """End the call after you've finished speaking."""
        await _hang_up()
        return "Ended."

    tools = []  # facility + intro: Klarra never hangs up, the caller does
    q_state = None
    if kind == "questionnaire":
        tools, q_state = build_questionnaire_tools(meta)
        instructions = questionnaire_instructions(meta)
        greet = f"Greet {meta['carer_name']} and check now is still OK."
    elif kind == "nurse":
        tools = [record_result, hang_up]
        instructions = nurse_instructions(meta)
        greet = f"Greet {meta['nurse_name']} and offer the shift."
    elif kind == "intro":
        instructions = intro_instructions(meta)
        greet = f"Warmly introduce yourself to {meta.get('nurse_name') or 'the carer'}."
    else:
        instructions = facility_instructions(meta)
        greet = "Greet the facility and give them the update."

    if TurnDetection is not None:
        rt = openai.realtime.RealtimeModel(
            voice="shimmer",
            speed=1.25,
            turn_detection=TurnDetection(
                type="semantic_vad",
                eagerness="medium",
                create_response=True,
                interrupt_response=True,
            ),
        )
    else:
        rt = openai.realtime.RealtimeModel(voice="shimmer", speed=1.25)
    session = AgentSession(llm=rt)

    if q_state is not None:
        @session.on("conversation_item_added")
        def _on_item(ev):
            try:
                item = ev.item
                text = getattr(item, "text_content", None)
                if text:
                    who = "Klarra" if item.role == "assistant" else "Carer"
                    q_state["transcript"].append(f"{who}: {text}")
            except Exception:
                pass

        def _q_disconnect(participant):
            if participant.identity == "callee":
                asyncio.create_task(asyncio.to_thread(finalise_questionnaire, meta, q_state))
        ctx.room.on("participant_disconnected", _q_disconnect)

        async def _timer():
            await asyncio.sleep(QUESTIONNAIRE_WRAP_SECONDS)
            if not q_state["finished"]:
                await session.generate_reply(instructions=(
                    "Time is nearly up. Finish the current question, then thank them, say the "
                    "rest will be picked up on another shift, give a full goodbye, then call finish_call."))
            await asyncio.sleep(QUESTIONNAIRE_MAX_SECONDS - QUESTIONNAIRE_WRAP_SECONDS)
            if not q_state["finished"]:
                await asyncio.to_thread(finalise_questionnaire, meta, q_state)
                await _hang_up()
        asyncio.create_task(_timer())

        import questionnaire as q
        sess = q.get_session(meta["session_id"])
        if sess:
            await asyncio.to_thread(q.update_session, sess, status="In Progress")

    await session.start(agent=Agent(instructions=instructions, tools=tools), room=ctx.room)
    await session.generate_reply(instructions=greet)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name=AGENT_NAME))
