"""
Read-only access to every table in Klarra's Supabase project, for her agents.

  query(sql)      run one SELECT (or WITH ... SELECT) through the klarra_query RPC.
                  The RPC runs as the klarra_reader role, which has SELECT only, so
                  nothing can be changed. 100 rows max, 8s timeout.
  schema_text()   a compact list of every table and column (cached for an hour),
                  given to Claude so it can write the SQL itself.
  TOOL            the Anthropic tool definition; call(args) runs it.

Tables added later are picked up automatically (default privileges + schema refresh).
"""

import json
import time
import logging

import db

logger = logging.getLogger("knightingale-data")

MAX_RESULT_CHARS = 8000
SCHEMA_TTL = 3600
HIDDEN_TYPES = {"vector", "tsvector"}
HIDDEN_COLUMNS = {"embedding", "search_vector"}

# Short notes so Claude knows what the main tables mean.
TABLE_NOTES = {
    "nurses": "carers (RN, EN, PCA, DSW). phone is E.164",
    "shifts": "rostered shifts. facility shifts have facility_id, NDIS shifts have participant_id. start_time/end_time are timestamptz (UTC), use AT TIME ZONE 'Australia/Melbourne'",
    "availability": "carer availability per date and block (Morning/Afternoon/Night); status assigned = booked",
    "facilities": "residential aged care sites",
    "participants": "NDIS participants (clients)",
    "users": "Bubble users export (carers and participants); quote column names with spaces",
    "recurring_shift_templates": "weekly recurring NDIS shifts. day_of_week 0=Monday",
    "progress_notes_search": "carers' progress notes per shift",
    "helpdesk_kb": "Klarra's saved answers taught by Paul",
    "helpdesk_messages": "SMS conversation log between Klarra and each carer",
    "helpdesk_questions": "each carer text and how it was handled",
    "health_profile": "participant health details (sensitive)",
    "risk_assessment": "participant risks (sensitive)",
    "home_safety": "participant home safety checklist",
    "daily_living": "participant preferences and daily living needs",
    "policy_chunks": "company policy documents, chunked",
    "sms_adhoc_offers": "shift offers Paul asked Klarra to text to carers",
    "shift_requests": "shift requests from facilities",
}

_schema_cache: tuple[float, str] | None = None


def query(sql: str):
    """Run one read-only query. Returns a list of rows, or {'error': ...}."""
    try:
        r = db.get_client().rpc("klarra_query", {"q": sql}).execute()
    except Exception as e:
        msg = getattr(e, "message", None) or str(e)
        logger.warning("klarra_query failed: %s | %s", msg, sql)
        return {"error": msg[:500]}
    rows = r.data if isinstance(r.data, list) else (r.data or [])
    for row in rows:
        if isinstance(row, dict):
            for k in list(row):
                if k in HIDDEN_COLUMNS:
                    row.pop(k)
    return rows


def schema_text() -> str:
    global _schema_cache
    if _schema_cache and time.time() - _schema_cache[0] < SCHEMA_TTL:
        return _schema_cache[1]
    hidden_types = ", ".join(f"'{t}'" for t in HIDDEN_TYPES)
    hidden_cols = ", ".join(f"'{c}'" for c in HIDDEN_COLUMNS)
    rows = query(
        "select table_name, string_agg(column_name || ' ' || "
        "case when data_type = 'USER-DEFINED' then udt_name "
        "when data_type = 'ARRAY' then 'array' else data_type end, ', ' "
        "order by ordinal_position) as cols "
        "from information_schema.columns where table_schema = 'public' "
        f"and udt_name not in ({hidden_types}) and column_name not in ({hidden_cols}) "
        "group by table_name order by table_name"
    )
    if isinstance(rows, dict):
        return _schema_cache[1] if _schema_cache else ""
    lines = []
    for r in rows:
        t = r["table_name"]
        note = f"  ({TABLE_NOTES[t]})" if t in TABLE_NOTES else ""
        lines.append(f"{t}{note}: {r['cols']}")
    text = "\n".join(lines)
    if text:
        _schema_cache = (time.time(), text)
    return text


TOOL = {
    "name": "query_database",
    "description": (
        "Run ONE read-only SQL SELECT against Knightingale's Supabase database (Postgres). "
        "Returns up to 100 rows as JSON. Use it to look up carers, shifts, rosters, "
        "participants, availability, offers, progress notes, policies and anything else "
        "in the schema. Never select embedding columns. Use ILIKE for names. Dates: "
        "shifts.start_time is UTC, convert with AT TIME ZONE 'Australia/Melbourne'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"sql": {"type": "string", "description": "A single SELECT or WITH query"}},
        "required": ["sql"],
    },
}


def call(args: dict) -> str:
    """Run the tool and return its result as a (size-capped) JSON string."""
    result = query((args or {}).get("sql", ""))
    text = json.dumps(result, default=str)
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + ' ... (truncated, narrow the query)'
    return text


def schema_prompt() -> str:
    s = schema_text()
    return ("\n\nDatabase tables (read-only, via query_database):\n" + s) if s else ""
