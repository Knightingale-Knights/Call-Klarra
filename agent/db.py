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


def get_candidate_pool(facility_slug: str, date: str, shift_type: str, role: str) -> list[dict]:
    """
    Return the eligible nurse pool for a shift, with decision attributes computed live.
    Hard filters (approved + available + correct role) are already applied inside the
    database function — every nurse returned is a valid option.
    """
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



# --- Shift request queue (Step C: inbound intake -> orchestrator handoff) ---

def create_shift_request(facility_id: int | None, callback_number: str,
                         date: str, shift_type: str, role: str,
                         source: str = "voice") -> int:
    """
    Write a shift request to the queue for the orchestrator to pick up and fill.
    Returns the new request id.
    In dev this is allowed through — it's the trigger for a test call.
    """
    client = get_client()
    resp = client.table("shift_requests").insert({
        "facility_id": facility_id,
        "facility_callback_number": callback_number,
        "date": date,
        "shift_type": shift_type,
        "role": role,
        "status": "pending",
        "source": source,
    }).execute()
    new_id = resp.data[0]["id"]
    logger.info("Created shift_request %s (%s %s %s)", new_id, date, shift_type, role)
    return new_id

# --- Orchestrator queue helpers (Step E) ---

def claim_next_request() -> dict | None:
    """
    Grab the oldest pending shift request and mark it 'working' so no other worker
    picks it up. Returns the request (joined with facility slug/name) or None.
    """
    client = get_client()
    pending = (
        client.table("shift_requests")
        .select("*, facilities(slug, name, complexity)")
        .eq("status", "pending")
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

# --- SMS sending (Twilio) ---

def send_sms(to: str, body: str) -> None:
    """Send an SMS via Twilio. In dev, redirect to KLARRA_DEV_PHONE (or block)."""
    if DEV:
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


def upsert_availability(nurse_id: int, date: str, shift_type: str) -> None:
    """Insert availability if not already present (unique on nurse+date+shift)."""
    if _blocked(f"upsert_availability nurse={nurse_id} {date}"):
        return
    client = get_client()
    existing = (client.table("availability").select("id")
                .eq("nurse_id", nurse_id).eq("date", date)
                .eq("shift_type", shift_type).limit(1).execute())
    if existing.data:
        return
    client.table("availability").insert({
        "nurse_id": nurse_id, "date": date, "shift_type": shift_type, "status": "pending",
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
