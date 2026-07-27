"""
Data layer — thin wrappers around the Supabase functions we built.

The agent never writes raw SQL. It calls these helpers, which call the database
functions (get_candidate_pool, match_learned_decisions) and the call_events table.
Keeping DB access here means the agent code stays about *conversation and decisions*,
not query plumbing.
"""

import os
import logging
from supabase import create_client, Client

logger = logging.getLogger("knightingale-agent.db")

# --- Safe mode ---
# KLARRA_MODE=dev blocks all writes + SMS (logged, no-op). Reads always pass through.
KLARRA_MODE = os.environ.get("KLARRA_MODE", "live").strip().strip('"').strip("'").lower()
DEV = KLARRA_MODE == "dev"


def _blocked(action: str) -> bool:
    """True if a write should be skipped because we're in dev mode."""
    if DEV:
        logger.warning("[DEV] blocked write: %s", action)
        return True
    return False


def pretty_date(d: str) -> str:
    """Format 'YYYY-MM-DD' as 'Sunday, June 21'. Returns input unchanged on failure."""
    from datetime import datetime
    try:
        dt = datetime.strptime(str(d)[:10], "%Y-%m-%d")
        return dt.strftime("%A, %B ") + str(dt.day)
    except Exception:
        return str(d)


_client: Client | None = None


def get_client() -> Client:
    """Lazily create one Supabase client, reused for the process lifetime."""
    global _client
    if _client is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_SERVICE_KEY"]
        _client = create_client(url, key)
        logger.info("Supabase client initialised")
    return _client


def _get_or_create_dev_nurse(role: str) -> dict | None:
    """Dev-only: find (or create) a real nurses-table row keyed on KLARRA_DEV_PHONE,
    so the hardcoded stand-in candidate has a genuine nurse_id — satisfies real
    foreign key constraints (e.g. sms_nurse_offers.nurse_id) instead of a fake
    sentinel that only worked for update-based writes."""
    dev_phone = os.environ.get("KLARRA_DEV_PHONE")
    if not dev_phone:
        return None
    client = get_client()
    existing = (
        client.table("nurses").select("id, first_name, role")
        .eq("phone", dev_phone).limit(1).execute()
    )
    if existing.data:
        row = existing.data[0]
        if row.get("role") != role:
            client.table("nurses").update({"role": role}).eq("id", row["id"]).execute()
            row["role"] = role
        return row
    logger.info("[DEV] creating stand-in dev nurse for %s", dev_phone)
    resp = client.table("nurses").insert({
        "first_name": "DevTest",
        "last_name": "Nurse",
        "phone": dev_phone,
        "role": role,
    }).execute()
    return resp.data[0]


def get_candidate_pool(facility_slug: str, date: str, shift_type: str, role: str) -> list[dict]:
    """
    Return the eligible nurse pool for a shift, with decision attributes computed live.
    Hard filters (approved + available + correct role) are already applied inside the
    database function — every nurse returned is a valid option.

    Dev mode: always returns a single stand-in nurse using KLARRA_DEV_PHONE — backed by
    a real nurses-table row (found or auto-created via _get_or_create_dev_nurse), so it
    satisfies real foreign key constraints, regardless of what's actually in Supabase
    for approvals/availability.
    """
    if DEV:
        dev_nurse = _get_or_create_dev_nurse(role)
        if dev_nurse:
            logger.info("[DEV] using stand-in nurse pool (nurse_id=%s)", dev_nurse["id"])
            return [{
                "nurse_id": dev_nurse["id"],
                "first_name": dev_nurse.get("first_name") or "DevTest",
                "phone": os.environ.get("KLARRA_DEV_PHONE"),
                "role": role,
                "reliability": 100,
            }]

    client = get_client()
    resp = client.rpc(
        "get_candidate_pool",
        {
            "p_facility_slug": facility_slug,
            "p_date": date,
            "p_shift_type": shift_type,
            "p_role": role,
        },
    ).execute()
    return resp.data or []


def match_learned_decisions(embedding: list[float], limit: int = 3) -> list[dict]:
    """
    Find prior rulings semantically similar to the current situation.
    Returns nearest matches with a 0–1 similarity score. Empty list = no precedent.
    """
    client = get_client()
    resp = client.rpc(
        "match_learned_decisions",
        {"p_embedding": embedding, "p_limit": limit},
    ).execute()
    return resp.data or []


def record_call_event(nurse_id: int, outcome: str, facility_id: int | None = None,
                      shift_date: str | None = None) -> None:
    """
    Log a call outcome to call_events. This is what feeds the reliability score, and
    also serves as the audit trail of who was called and what happened.
    Valid outcomes: accepted, declined, no_answer, completed, no_show, cancelled.
    """
    if _blocked(f"record_call_event nurse={nurse_id} {outcome}"):
        return
    client = get_client()
    client.table("call_events").insert({
        "nurse_id": nurse_id,
        "outcome": outcome,
        "facility_id": facility_id,
        "shift_date": shift_date,
    }).execute()
    logger.info("Recorded call event: nurse %s -> %s", nurse_id, outcome)


# --- Embeddings + saving rulings (Step 13: the learning loop) ---

import openai as openai_sdk  # the official OpenAI client, for embeddings

_openai_client = None


def _get_openai():
    global _openai_client
    if _openai_client is None:
        _openai_client = openai_sdk.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client


def embed_situation(text: str) -> list[float]:
    """
    Turn a plain-language situation description into a 1536-dim vector using
    text-embedding-3-small. This is the 'meaning fingerprint' used to find similar
    past rulings (and to store new ones).
    """
    resp = _get_openai().embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return resp.data[0].embedding


def insert_learned_decision(situation_text: str, ruling: str,
                            tags: dict | None = None) -> None:
    """
    Save a ruling Paul has given on a tricky situation, with its embedding, so the
    agent can reason from it next time. This is the ONLY place the agent 'learns' —
    and every row traces back to a real decision Paul made.
    """
    if _blocked(f"insert_learned_decision {situation_text[:40]}"):
        return
    client = get_client()
    embedding = embed_situation(situation_text)
    client.table("learned_decisions").insert({
        "situation_text": situation_text,
        "embedding": embedding,
        "ruling": ruling,
        "tags": tags or {},
    }).execute()
    logger.info("Saved learned decision: %s", situation_text[:60])


# --- Facility identification by caller number (Step 15) ---

def list_facilities() -> list[dict]:
    """All facilities (id, name, slug) — used to let a caller name a different
    destination facility (e.g. Collins booking on behalf of another site)."""
    client = get_client()
    r = client.table("facilities").select("id, name, slug").execute()
    return r.data or []


def facility_by_phone(phone: str) -> dict | None:
    """
    Look up which facility a phone number belongs to, via facility_phones.
    Returns the facility row (id, name, slug, complexity) or None if the number
    is not known. A facility may have many numbers; any of them resolves here.
    """
    client = get_client()
    resp = (
        client.table("facility_phones")
        .select("facility_id, facilities(id, name, slug, complexity)")
        .eq("phone", phone)
        .limit(1)
        .execute()
    )
    if not resp.data:
        return None
    row = resp.data[0]
    return row.get("facilities")


def first_facility() -> dict | None:
    """Dev helper: return any one facility, used as a stand-in for unknown callers."""
    client = get_client()
    resp = client.table("facilities").select("id, name, slug, complexity").limit(1).execute()
    return resp.data[0] if resp.data else None


# --- Afterhours unknown-caller threads ---

def get_afterhours_thread(phone: str) -> dict | None:
    client = get_client()
    r = client.table("afterhours_threads").select("*").eq("phone", phone).limit(1).execute()
    return r.data[0] if r.data else None


def save_afterhours_thread(phone: str, messages: list, done: bool = False,
                           summarised: bool = False) -> None:
    """Upsert the running transcript for an unknown caller."""
    client = get_client()
    client.table("afterhours_threads").upsert({
        "phone": phone,
        "messages": messages,
        "done": done,
        "summarised": summarised,
        "updated_at": "now()",
    }).execute()



# --- Shift request queue (Step C: inbound intake -> orchestrator handoff) ---

def create_shift_request(facility_id: int | None, callback_number: str,
                         date: str, shift_type: str, role: str,
                         source: str = "voice",
                         start_time: str | None = None,
                         end_time: str | None = None) -> int:
    client = get_client()
    resp = client.table("shift_requests").insert({
        "facility_id": facility_id,
        "facility_callback_number": callback_number,
        "date": date,
        "shift_type": shift_type,
        "role": role,
        "status": "pending",
        "source": source,
        "start_time": start_time,
        "end_time": end_time,
    }).execute()
    new_id = resp.data[0]["id"]
    logger.info("Created shift_request %s (%s %s %s)", new_id, date, shift_type, role)
    return new_id

# --- Orchestrator queue helpers (Step E) ---

def claim_next_request() -> dict | None:
    """
    Grab the oldest pending VOICE-lane shift request and mark it 'working' so no
    other worker picks it up. SMS-sourced requests are claimed separately by
    claim_next_sms_request() so the two orchestrators never race on the same row.
    Returns the request (joined with facility slug/name) or None.
    """
    client = get_client()
    pending = (
        client.table("shift_requests")
        .select("*, facilities(slug, name, complexity)")
        .eq("status", "pending")
        .neq("source", "sms")
        .order("created_at")
        .limit(1)
        .execute()
    )
    if not pending.data:
        return None
    req = pending.data[0]
    if _blocked(f"claim_next_request mark working id={req['id']}"):
        return req
    client.table("shift_requests").update(
        {"status": "working", "updated_at": "now()"}
    ).eq("id", req["id"]).execute()
    return req


def assign_availability(nurse_id: int, date: str, shift_type: str) -> bool:
    """Conditionally flip this nurse's availability row to 'assigned' for date+shift —
    ONLY if it's still 'pending'. Returns True if this call won the race, False if
    someone already took it (or no row exists)."""
    client = get_client()
    resp = (
        client.table("availability")
        .update({"status": "assigned"})
        .eq("nurse_id", nurse_id).eq("date", date).eq("shift_type", shift_type)
        .eq("status", "pending")
        .execute()
    )
    won = bool(resp.data)
    logger.info("assign_availability nurse=%s %s %s -> %s", nurse_id, date, shift_type, won)
    return won


def next_rotation() -> int:
    """Atomically increment the global rotation counter and return its new value."""
    client = get_client()
    resp = client.rpc("increment_rotation_counter").execute()
    return resp.data


def mark_request_filled(request_id: int, nurse_id: int) -> None:
    if _blocked(f"mark_request_filled id={request_id} nurse={nurse_id}"):
        return
    client = get_client()
    client.table("shift_requests").update(
        {"status": "filled", "filled_by_nurse_id": nurse_id, "updated_at": "now()"}
    ).eq("id", request_id).execute()
    logger.info("Request %s filled by nurse %s", request_id, nurse_id)


def mark_request_unfilled(request_id: int) -> None:
    if _blocked(f"mark_request_unfilled id={request_id}"):
        return
    client = get_client()
    client.table("shift_requests").update(
        {"status": "unfilled", "updated_at": "now()"}
    ).eq("id", request_id).execute()
    logger.info("Request %s unfilled — no nurse found", request_id)


def mark_request_done_dev(request_id: int) -> None:
    """Dev-only: mark a test request done so the orchestrator won't re-claim and
    re-call. Bypasses the write guard ON PURPOSE — only runs when DEV is true."""
    if not DEV:
        return
    client = get_client()
    client.table("shift_requests").update(
        {"status": "dev_done", "updated_at": "now()"}
    ).eq("id", request_id).execute()
    logger.info("[DEV] request %s marked dev_done", request_id)


def set_dev_outcome(request_id: int, outcome: str) -> None:
    """Dev-only: store the nurse's outcome on the request row so the orchestrator
    can read it back and reply accurately. Bypasses the guard on purpose."""
    if not DEV:
        return
    client = get_client()
    client.table("shift_requests").update(
        {"status": f"dev_{outcome}", "updated_at": "now()"}
    ).eq("id", request_id).execute()
    logger.info("[DEV] request %s outcome -> %s", request_id, outcome)


def get_dev_outcome(request_id: int) -> str | None:
    """Dev-only: read back the stored outcome (accepted/declined) if present."""
    client = get_client()
    r = (client.table("shift_requests").select("status")
         .eq("id", request_id).limit(1).execute())
    if not r.data:
        return None
    st = r.data[0]["status"]
    if st in ("dev_accepted", "dev_declined"):
        return st.removeprefix("dev_")
    return None


def set_awaiting_approval(request_id: int, nurse_id: int, nurse_name: str,
                          reason: str) -> None:
    """Park a request pending Paul's YES/NO. Stores the chosen nurse + reason.
    Allowed in dev too (it's how the gate is tested)."""
    client = get_client()
    client.table("shift_requests").update({
        "status": "awaiting_approval",
        "approval_nurse_id": nurse_id,
        "approval_nurse_name": nurse_name,
        "approval_reason": reason,
        "updated_at": "now()",
    }).eq("id", request_id).execute()
    logger.info("Request %s awaiting approval (nurse %s)", request_id, nurse_name)


def get_awaiting_approval() -> dict | None:
    """Return the most recent request parked awaiting approval, with facility join."""
    client = get_client()
    r = (client.table("shift_requests")
         .select("*, facilities(slug, name, complexity)")
         .eq("status", "awaiting_approval")
         .order("updated_at", desc=True)
         .limit(1)
         .execute())
    return r.data[0] if r.data else None


def resolve_approval(request_id: int, approved: bool, nurse_id: int | None) -> None:
    """Apply Paul's decision: approved -> filled, else unfilled."""
    client = get_client()
    if approved:
        payload = {"status": "filled", "filled_by_nurse_id": nurse_id,
                   "updated_at": "now()"}
    else:
        payload = {"status": "unfilled", "updated_at": "now()"}
    client.table("shift_requests").update(payload).eq("id", request_id).execute()
    logger.info("Request %s approval resolved -> %s", request_id,
                "filled" if approved else "unfilled")


# --- SMS shift-state queue (SMS nurse-offer workflow) ---

def claim_next_sms_request() -> dict | None:
    """
    Grab the oldest pending SMS-lane shift request and mark it 'working'. Mirrors
    claim_next_request() but filtered to source='sms', so the SMS orchestrator and
    the voice orchestrator never claim the same row.

    Always marks 'working' for real, even in dev — this is a claim lock, not a
    real-world side effect. Leaving it blocked caused the same pending row to be
    re-claimed and reprocessed every poll cycle instead of just once.
    """
    client = get_client()
    pending = (
        client.table("shift_requests")
        .select("*, facilities(slug, name, complexity)")
        .eq("status", "pending")
        .eq("source", "sms")
        .order("created_at")
        .limit(1)
        .execute()
    )
    if not pending.data:
        return None
    req = pending.data[0]
    client.table("shift_requests").update(
        {"status": "working", "updated_at": "now()"}
    ).eq("id", req["id"]).execute()
    return req


def create_sms_offers(shift_request_id: int, ranked_pool: list[dict]) -> None:
    """
    Set up the SMS offer cascade for a shift request: one sms_shift_state header row
    (status='offering'), and one sms_nurse_offers row per ranked nurse (status='pending',
    in rank order starting at 1).

    Always writes for real, even in dev — these tables only track the SMS test/offer
    flow itself (not real nurses/facilities/shifts), so there's nothing to protect by
    blocking them; blocking them would just break dev testing entirely.

    Idempotent: if a cascade already exists for this request (e.g. a crashed or
    reprocessed attempt), skips re-creating it instead of erroring on the duplicate key.
    """
    client = get_client()
    existing = (
        client.table("sms_shift_state")
        .select("shift_request_id")
        .eq("shift_request_id", shift_request_id)
        .limit(1)
        .execute()
    )
    if existing.data:
        logger.info("SMS offer cascade already exists for request %s, skipping create",
                    shift_request_id)
        return
    client.table("sms_shift_state").insert({
        "shift_request_id": shift_request_id,
        "status": "offering",
    }).execute()
    rows = [
        {
            "shift_request_id": shift_request_id,
            "nurse_id": n["nurse_id"],
            "rank_position": i,
            "status": "pending",
        }
        for i, n in enumerate(ranked_pool, 1)
    ]
    if rows:
        client.table("sms_nurse_offers").insert(rows).execute()
    logger.info("Created SMS offer cascade for request %s (%d nurses)",
                shift_request_id, len(rows))


def get_next_pending_offer(shift_request_id: int) -> dict | None:
    """Return the lowest rank_position offer still 'pending' for this request, or None
    if the pool is exhausted."""
    client = get_client()
    r = (
        client.table("sms_nurse_offers")
        .select("*, nurses(id, first_name, phone)")
        .eq("shift_request_id", shift_request_id)
        .eq("status", "pending")
        .order("rank_position")
        .limit(1)
        .execute()
    )
    return r.data[0] if r.data else None


def mark_offer(offer_id: str, status: str, **timestamps) -> None:
    """
    Update one sms_nurse_offers row's status, plus any of offered_at / alert_called_at /
    replied_at passed in timestamps (as ISO strings, or 'now()' literal).
    Valid status: pending, offered, alerted, accepted, declined, no_reply, skipped.
    Always writes for real, even in dev — see create_sms_offers.
    """
    client = get_client()
    payload = {"status": status, **timestamps}
    client.table("sms_nurse_offers").update(payload).eq("id", offer_id).execute()
    logger.info("Offer %s -> %s", offer_id, status)


def get_offer(offer_id: str) -> dict | None:
    """Read back a single sms_nurse_offers row by id — used to poll for a nurse's
    reply after an offer/alert without re-querying the whole pending list."""
    client = get_client()
    r = client.table("sms_nurse_offers").select("*").eq("id", offer_id).limit(1).execute()
    return r.data[0] if r.data else None


def get_sms_state(shift_request_id: int) -> dict | None:
    """Read the sms_shift_state header row for a shift request."""
    client = get_client()
    r = (client.table("sms_shift_state").select("*")
         .eq("shift_request_id", shift_request_id).limit(1).execute())
    return r.data[0] if r.data else None


def mark_sms_state(shift_request_id: int, status: str | None = None, **fields) -> None:
    """Update the sms_shift_state row for a shift request. Pass status and/or any of
    confirmed_nurse_id / admin_reminder_count / admin_notified_at / admin_approved_at.
    Always writes for real, even in dev — see create_sms_offers."""
    client = get_client()
    payload = {"updated_at": "now()", **fields}
    if status is not None:
        payload["status"] = status
    client.table("sms_shift_state").update(payload).eq(
        "shift_request_id", shift_request_id
    ).execute()
    logger.info("SMS state %s -> %s", shift_request_id, status)


def place_alert_call(phone: str) -> None:
    """Ring a nurse's phone as a nudge (no conversation) — hangs up as soon as it's
    answered. In dev, redirects to KLARRA_DEV_PHONE like send_sms does."""
    to = phone
    if DEV and to not in dev_testers():
        dev_to = os.environ.get("KLARRA_DEV_PHONE")
        if not dev_to:
            logger.warning("[DEV] blocked alert call to %s", to)
            return
        logger.warning("[DEV] redirect alert call %s -> %s", to, dev_to)
        to = dev_to
    from twilio.rest import Client as TwilioClient
    tw = TwilioClient(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
    tw.calls.create(
        to=to,
        from_=os.environ["TWILIO_PHONE_NUMBER"],
        twiml="<Response><Hangup/></Response>",
    )
    logger.info("Placed alert call to %s", to)


def get_active_offer_by_phone(phone: str) -> dict | None:
    """Find this nurse's most recent unresolved (offered/alerted) SMS offer, if any —
    used by sms_webhook to route a YES/NO reply to the right offer row."""
    client = get_client()
    nurse = client.table("nurses").select("id, first_name").eq("phone", phone).limit(1).execute()
    if not nurse.data:
        return None
    r = (
        client.table("sms_nurse_offers")
        .select("*, shift_requests(id, facility_id, facility_callback_number, date, shift_type)")
        .eq("nurse_id", nurse.data[0]["id"])
        .in_("status", ["offered", "alerted"])
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if not r.data:
        return None
    row = r.data[0]
    row["nurse_first_name"] = nurse.data[0]["first_name"]
    return row


def get_pending_admin_approval() -> dict | None:
    """Most recent sms_shift_state row awaiting Paul's OK, if any — used by
    sms_webhook to route his reply without mistaking it for a shift request."""
    client = get_client()
    r = (
        client.table("sms_shift_state")
        .select("*, shift_requests(id, facility_id, facility_callback_number, date, shift_type)")
        .eq("status", "pending_admin_approval")
        .order("admin_notified_at", desc=True)
        .limit(1)
        .execute()
    )
    return r.data[0] if r.data else None


def get_sms_state(shift_request_id: int) -> dict | None:
    """Return the sms_shift_state row for this request, or None."""
    client = get_client()
    r = (client.table("sms_shift_state").select("*")
         .eq("shift_request_id", shift_request_id).limit(1).execute())
    return r.data[0] if r.data else None


def mark_sms_state(shift_request_id: int, status: str, **fields) -> None:
    """Update the sms_shift_state row's status, plus any other columns passed in
    fields (e.g. confirmed_nurse_id, admin_notified_at, admin_reminder_count)."""
    if _blocked(f"mark_sms_state request={shift_request_id} -> {status}"):
        return
    client = get_client()
    payload = {"status": status, "updated_at": "now()", **fields}
    client.table("sms_shift_state").update(payload).eq(
        "shift_request_id", shift_request_id
    ).execute()
    logger.info("SMS state for request %s -> %s", shift_request_id, status)


# --- SMS sending (Twilio) ---

def dev_testers() -> set:
    """Recognised tester numbers in dev (comma-separated KLARRA_DEV_PHONES,
    plus the primary KLARRA_DEV_PHONE)."""
    raw = os.environ.get("KLARRA_DEV_PHONES", "")
    s = {p.strip() for p in raw.split(",") if p.strip()}
    primary = os.environ.get("KLARRA_DEV_PHONE")
    if primary:
        s.add(primary)
    return s


def send_sms(to: str, body: str) -> None:
    """Send an SMS via Twilio. In dev, allow sends to known testers (to their own
    number); redirect anything else to the primary dev phone (or block)."""
    if DEV and to not in dev_testers():
        dev_to = os.environ.get("KLARRA_DEV_PHONE")
        if not dev_to:
            logger.warning("[DEV] blocked SMS to %s: %s", to, body[:60])
            return
        logger.warning("[DEV] redirect SMS %s -> %s", to, dev_to)
        to = dev_to
    from twilio.rest import Client as TwilioClient
    tw = TwilioClient(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
    tw.messages.create(to=to, from_=os.environ["TWILIO_PHONE_NUMBER"], body=body)
    logger.info("Sent SMS to %s", to)


def place_alert_call(to: str) -> None:
    """Place a short alert call via Twilio that hangs up immediately once answered —
    a ring-based nudge to check the SMS offer just sent. Same DEV redirect as send_sms."""
    if DEV and to not in dev_testers():
        dev_to = os.environ.get("KLARRA_DEV_PHONE")
        if not dev_to:
            logger.warning("[DEV] blocked alert call to %s", to)
            return
        logger.warning("[DEV] redirect alert call %s -> %s", to, dev_to)
        to = dev_to
    from twilio.rest import Client as TwilioClient
    tw = TwilioClient(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
    tw.calls.create(
        to=to,
        from_=os.environ["TWILIO_PHONE_NUMBER"],
        twiml="<Response><Hangup/></Response>",
    )
    logger.info("Placed alert call to %s", to)

# --- Bubble sync helpers ---

FACILITY_NAME_TO_SLUG = {
    "Mclean Lodge": "mclean_lodge",
    "Williamstown Hostel": "williamstown",
    "Ron Conn": "ron_con",
    "Angus Martin": "angus_martin",
    "Port Melbourne": "port_melbourne",
    "Eunice Seddon": "eunice_seddon",
}


def upsert_nurse(bubble_id: str, first_name: str, last_name: str, phone: str,
                 role: str, address: str | None) -> int:
    """Insert or update a nurse keyed by their Bubble _id. Returns nurse id."""
    if _blocked(f"upsert_nurse {bubble_id}"):
        return -1
    client = get_client()
    existing = client.table("nurses").select("id").eq("bubble_user_id", bubble_id).limit(1).execute()
    payload = {
        "first_name": first_name, "last_name": last_name, "phone": phone,
        "role": role, "address": address, "bubble_user_id": bubble_id,
    }
    if existing.data:
        nid = existing.data[0]["id"]
        client.table("nurses").update(payload).eq("id", nid).execute()
        return nid
    resp = client.table("nurses").insert(payload).execute()
    return resp.data[0]["id"]


def set_nurse_approvals(nurse_id: int, slugs: list[str]) -> None:
    """Replace a nurse's facility approvals with the given slugs."""
    if _blocked(f"set_nurse_approvals nurse={nurse_id}"):
        return
    client = get_client()
    # clear existing
    client.table("nurse_facility_approvals").delete().eq("nurse_id", nurse_id).execute()
    for slug in slugs:
        fac = client.table("facilities").select("id").eq("slug", slug).limit(1).execute()
        if fac.data:
            client.table("nurse_facility_approvals").insert({
                "nurse_id": nurse_id, "facility_id": fac.data[0]["id"],
            }).execute()


def nurse_id_by_bubble(bubble_id: str) -> int | None:
    client = get_client()
    r = client.table("nurses").select("id").eq("bubble_user_id", bubble_id).limit(1).execute()
    return r.data[0]["id"] if r.data else None


def upsert_availability(nurse_id: int, date: str, shift_type: str, bubble_id: str | None = None) -> None:
    """Insert availability if not already present (unique on nurse+date+shift).
    If it already exists, backfill bubble_id when missing."""
    if _blocked(f"upsert_availability nurse={nurse_id} {date}"):
        return
    client = get_client()
    existing = (client.table("availability").select("id,bubble_id")
                .eq("nurse_id", nurse_id).eq("date", date)
                .eq("shift_type", shift_type).limit(1).execute())
    if existing.data:
        row = existing.data[0]
        if bubble_id and not row.get("bubble_id"):
            client.table("availability").update({"bubble_id": bubble_id}).eq("id", row["id"]).execute()
        return
    client.table("availability").insert({
        "nurse_id": nurse_id, "date": date, "shift_type": shift_type, "status": "pending",
        "bubble_id": bubble_id,
    }).execute()

# --- Shift history sync ---

LOCATION_ID_TO_SLUG = {
    "1714536331477x218496158382794920": "mclean_lodge",
    "1725011874725x652462711584855800": "williamstown",
    "1736306652404x854626961349243600": "ron_con",
    "1740520501450x744150619674484000": "angus_martin",
    "1743477412156x587548361481612400": "port_melbourne",
    "1764815902993x496620108715589700": "eunice_seddon",
}


def facility_id_by_slug(slug: str) -> int | None:
    client = get_client()
    r = client.table("facilities").select("id").eq("slug", slug).limit(1).execute()
    return r.data[0]["id"] if r.data else None


def upsert_shift(bubble_shift_id: str, nurse_id: int, facility_id: int,
                 date: str, shift_type: str, start_time: str, end_time: str,
                 status: str) -> None:
    """Insert a worked shift if not already present (keyed by bubble shift id stored
    nowhere yet — so we dedupe on nurse+facility+date+start)."""
    if _blocked(f"upsert_shift nurse={nurse_id} {date}"):
        return
    client = get_client()
    existing = (client.table("shifts").select("id")
                .eq("nurse_id", nurse_id).eq("facility_id", facility_id)
                .eq("date", date).eq("start_time", start_time).limit(1).execute())
    if existing.data:
        return
    client.table("shifts").insert({
        "nurse_id": nurse_id, "facility_id": facility_id, "date": date,
        "shift_type": shift_type, "start_time": start_time, "end_time": end_time,
        "status": status,
    }).execute()
