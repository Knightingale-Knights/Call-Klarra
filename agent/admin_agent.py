"""
Admin agent: lets Paul talk to Klarra in plain language by SMS.

Replaces the fixed three-command parser for Paul's number. Claude is given a small
set of tools and decides what to call, so Paul never has to follow a script:

  check_availability   who is free (any role, any/all blocks, any date)
  check_nurse          is one named carer free
  offer_shift          text one OR MORE named carers to ask if they can work a shift
  check_replies        who has said yes / no / nothing yet
  create_shift         create a facility shift in Bubble (and Supabase) for a named carer
  create_ndis_shift    create an NDIS shift (participant + NDIS Pricing item)
  not_an_admin_request hand the message back to the normal routing

Conversation memory: the last few turns are kept per phone in admin_threads for
THREAD_MINUTES, so "Which Maria?" -> "Maria Santos" works.

Group offers share a batch_id on sms_adhoc_offers. When every carer in a batch has
replied, sms_webhook texts Paul a one-line summary via batch_summary().

KLARRA_MODE is respected: send_sms already redirects in dev and blocks nurses in mid.
In mid, offer_shift refuses outright so Klarra never claims to have texted someone
she didn't.
"""

import os
import json
import uuid
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import anthropic
import requests

import db

logger = logging.getLogger("knightingale-admin-agent")

MODEL = "claude-sonnet-4-6"
ROLES = ["RN", "EN", "PCA", "DSW"]
BLOCKS = ["Morning", "Afternoon", "Night"]
MAX_STEPS = 8
THREAD_MINUTES = 30
MAX_HISTORY = 10  # messages kept per thread (user + assistant)


def _now_melb() -> datetime:
    return datetime.now(ZoneInfo("Australia/Melbourne"))


def _system() -> str:
    now = _now_melb()
    return (
        "You are Klarra, the scheduling assistant for Knightingale, a Melbourne aged "
        "care and NDIS staffing agency. You are texting with Paul, the director. "
        f"Today is {now.strftime('%A %d %B %Y')} ({now.strftime('%Y-%m-%d')}), "
        "Australia/Melbourne time. Resolve words like tomorrow, Friday, next Tuesday "
        "to a YYYY-MM-DD date yourself.\n\n"
        "Reply as an SMS: plain text, short, no markdown, never use em dashes. Paul "
        "writes casually, so interpret loosely. Never invent names or availability, "
        "always use the tools.\n\n"
        "Roles: RN, EN, PCA, DSW (\"EN's\" means EN). Shift blocks: AM or morning = "
        "Morning, PM or arvo or afternoon = Afternoon, NS or night or overnight = "
        "Night.\n\n"
        "What you can do:\n"
        "1. Say who is available (check_availability). If Paul gives no role, check "
        "all roles. If he gives no block, check all blocks and show each person's "
        "blocks. check_availability returns reply_text already formatted: send "
        "reply_text exactly as given, nothing added or removed. Questions about "
        "availability never send any texts.\n"
        "2. Say whether one named carer is available (check_nurse).\n"
        "3. Text one or more named carers to ask if they can work a shift "
        "(offer_shift). You need the carer names, the site, the date and the block. "
        "If something is missing, ask for just that, one question. Texts are real, "
        "so only call offer_shift when Paul clearly asked you to text or ask carers.\n"
        "4. Report replies (check_replies) when Paul asks who has answered.\n"
        "If a name matches several carers, ask which one. If a tool returns an error, "
        "tell Paul plainly what went wrong.\n"
        "After offer_shift succeeds, confirm briefly who was texted and that you will "
        "tell him as replies come in.\n"
        "5. Create a shift in Bubble for a named carer (create_shift). Needs carer, "
        "site, date and block. Only call it when Paul clearly asks you to create or "
        "book the shift. Williamstown has several codes: D6, D7, D9 are Morning, A2 "
        "and A3 are Afternoon, Night has one. If the tool says a choice is needed, "
        "ask Paul which code, one question. When create_shift succeeds, send "
        "reply_text exactly as given. Facility shifts only use the standard times "
        "for each site, not custom times.\n"
        "6. Create an NDIS shift for a named carer and a named participant "
        "(create_ndis_shift). Needs carer, participant, date, start time and end time "
        "(24 hour HH:MM, any times). It also needs item_prefix (01 self-care or 04 "
        "community access) and day_kind (daytime, evening, night, saturday, sunday or "
        "public_holiday). ALWAYS ask Paul for the item prefix and the time of day "
        "unless he has stated them, for example 'shift is 04 daytime', 'shift is 01 "
        "evening', 'shift is 04 sunday', 'shift is 01 night'. Ask for both in one "
        "short question. Daytime is 0600 to 2000, evening 2000 to 2400, night 2400 to "
        "0600. Never guess them. When it succeeds, send reply_text exactly as given. "
        "DSW carers work NDIS shifts. A named site means a facility shift (5), a "
        "named participant means an NDIS shift (6).\n"
        "If the message is not addressed to you as an assistant, for example a "
        "facility style request to fill a shift through the normal system such as "
        "'EN morning shift tomorrow at Port Melbourne' stated as a need rather than a "
        "question about who is free, call not_an_admin_request."
    )


TOOLS = [
    {
        "name": "check_availability",
        "description": (
            "List carers who are free on a date. Optional role (RN/EN/PCA/DSW) and "
            "optional shift block. Omit role to check all roles, omit shift_type to "
            "check all blocks. Excludes anyone already rostered that day. Not "
            "facility-specific."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "role": {"type": "string", "enum": ROLES},
                "shift_type": {"type": "string", "enum": BLOCKS},
            },
            "required": ["date"],
        },
    },
    {
        "name": "check_nurse",
        "description": "Check which blocks one named carer is free on a date.",
        "input_schema": {
            "type": "object",
            "properties": {
                "nurse_name": {"type": "string"},
                "date": {"type": "string", "description": "YYYY-MM-DD"},
            },
            "required": ["nurse_name", "date"],
        },
    },
    {
        "name": "offer_shift",
        "description": (
            "Text one or more named carers asking if they can work a shift. Sends "
            "real SMS. Their YES/NO replies are tracked and reported to Paul. Sends "
            "nothing if any name is ambiguous or unknown."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "nurse_names": {"type": "array", "items": {"type": "string"}},
                "facility": {"type": "string", "description": "Site name as Paul said it"},
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "shift_type": {"type": "string", "enum": BLOCKS},
            },
            "required": ["nurse_names", "facility", "date", "shift_type"],
        },
    },
    {
        "name": "create_shift",
        "description": (
            "Create a shift in Bubble (visible to carers in the app) for one named "
            "carer at a site, using that site's standard times. Writes real data. "
            "Refuses if the carer already has a shift that day."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "nurse_name": {"type": "string"},
                "facility": {"type": "string", "description": "Site name as Paul said it"},
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "shift_type": {"type": "string", "enum": BLOCKS},
                "shift_code": {
                    "type": "string",
                    "description": "Williamstown only: D6, D7, D9, A2 or A3",
                },
            },
            "required": ["nurse_name", "facility", "date", "shift_type"],
        },
    },
    {
        "name": "create_ndis_shift",
        "description": (
            "Create an NDIS shift in Bubble for one named carer and one named "
            "participant, priced from the NDIS Pricing table. No unpaid break is "
            "deducted. Writes real data. Only call once item_prefix and day_kind are "
            "known."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "nurse_name": {"type": "string"},
                "participant": {"type": "string", "description": "Participant name as Paul said it"},
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "start_time": {"type": "string", "description": "24 hour HH:MM"},
                "end_time": {"type": "string", "description": "24 hour HH:MM"},
                "item_prefix": {"type": "string", "enum": ["01", "04"]},
                "day_kind": {
                    "type": "string",
                    "enum": ["daytime", "evening", "night", "saturday", "sunday", "public_holiday"],
                },
            },
            "required": ["nurse_name", "participant", "date", "start_time",
                         "end_time", "item_prefix", "day_kind"],
        },
    },
    {
        "name": "check_replies",
        "description": (
            "Show the status (accepted / declined / waiting) of shift offers texted "
            "to carers. Optionally filter by shift date. Defaults to recent offers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"date": {"type": "string", "description": "YYYY-MM-DD"}},
        },
    },
    {
        "name": "not_an_admin_request",
        "description": (
            "Call this if the message is not a request for you to do something here "
            "and should be handled by the normal shift-request routing instead."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]


# --- helpers ---------------------------------------------------------------

def _valid_date(d) -> bool:
    try:
        datetime.strptime(str(d), "%Y-%m-%d")
        return True
    except Exception:
        return False


def _resolve_nurse(name: str) -> tuple[dict | None, dict | None]:
    """Return (nurse, None) on a confident match, else (None, error_dict)."""
    nurse, candidates = db.find_nurse_by_name(name)
    if candidates:
        return None, {
            "name": name,
            "problem": "ambiguous",
            "matches": [f"{c['first_name']} {c['last_name']}" for c in candidates],
        }
    if not nurse:
        return None, {"name": name, "problem": "not_found"}
    return nurse, None


def offer_message(nurse: dict, facility: dict, date: str, shift_type: str) -> str:
    return (
        f"Hi {nurse['first_name']}, I've got a shift at {facility['name']} "
        f"on {db.short_date(date)} ({shift_type}). Please reply YES if you would "
        f"like it. Please reply NO if you would prefer to pass. Thank you"
    )


# --- tools -----------------------------------------------------------------

BLOCK_LABEL = {"Morning": "AM", "Afternoon": "PM", "Night": "NS"}


def _format_availability(by_role: dict, blocks: list) -> str:
    """AM:/PM:/NS: sections, one '- name' line per carer. First names only, with a
    last initial added when two different carers share a first name. A role heading
    is added only when more than one role is shown."""
    everyone = {}
    for people in by_role.values():
        for p in people:
            everyone[(p["first_name"], p["last_name"])] = p
    first_counts = {}
    for first, _last in everyone:
        first_counts[first] = first_counts.get(first, 0) + 1

    def display(p):
        first = p["first_name"]
        if first_counts.get(first, 0) > 1 and p.get("last_name"):
            return f"{first} {p['last_name'][0]}."
        return first

    sections = []
    for role, people in by_role.items():
        lines = []
        for b in blocks:
            names = sorted(display(p) for p in people if b in p["blocks"])
            lines.append(f"{BLOCK_LABEL[b]}:")
            lines.extend(f"- {n}" for n in (names or ["none"]))
            lines.append("")
        text = "\n".join(lines).strip()
        sections.append(f"{role}\n{text}" if len(by_role) > 1 else text)
    return "\n\n".join(sections)


def tool_check_availability(date, role=None, shift_type=None, **_):
    if not _valid_date(date):
        return {"error": "bad date"}
    roles = [role.upper()] if role else ROLES
    blocks = [shift_type] if shift_type else BLOCKS
    if any(r not in ROLES for r in roles) or any(b not in BLOCKS for b in blocks):
        return {"error": "bad role or shift_type"}
    out = {}
    for r in roles:
        people = {}
        for b in blocks:
            for n in db.get_available_nurses(date, b, r):
                p = people.setdefault(n["nurse_id"], {
                    "first_name": n["first_name"],
                    "last_name": n["last_name"],
                    "blocks": [],
                })
                p["blocks"].append(b)
        out[r] = list(people.values())
    return {"date": date, "reply_text": _format_availability(out, blocks)}


def _nurse_blocks(nurse_id: int, date: str) -> list[str]:
    """Blocks this carer is free for on a date: pending availability rows, and none
    at all if they already have any shift that day. Kept here so the agent does not
    depend on a db.py helper."""
    client = db.get_client()
    avail = (client.table("availability").select("shift_type")
             .eq("nurse_id", nurse_id).eq("date", date).eq("status", "pending")
             .execute().data or [])
    blocks = [r["shift_type"] for r in avail]
    if not blocks:
        return []
    clash = (client.table("shifts").select("id")
             .eq("nurse_id", nurse_id).eq("date", date).limit(1).execute().data)
    return [] if clash else blocks


def tool_check_nurse(nurse_name, date, **_):
    if not _valid_date(date):
        return {"error": "bad date"}
    nurse, err = _resolve_nurse(nurse_name)
    if err:
        return err
    blocks = _nurse_blocks(nurse["id"], date)
    return {"carer": f"{nurse['first_name']} {nurse['last_name']}",
            "date": date, "available_blocks": blocks}


def tool_offer_shift(nurse_names, facility, date, shift_type, **_):
    if db.MID:
        return {"error": "KLARRA_MODE is mid, so carers cannot be contacted. Nothing sent."}
    if not _valid_date(date) or shift_type not in BLOCKS:
        return {"error": "bad date or shift_type"}
    if not nurse_names:
        return {"error": "no carers named"}

    fac = db.find_facility_by_name(facility)
    if not fac:
        return {"error": f"could not match a site called '{facility}'"}

    # Resolve everyone first so a single bad name sends nothing at all.
    resolved, problems = [], []
    for name in nurse_names:
        nurse, err = _resolve_nurse(name)
        if err:
            problems.append(err)
        else:
            resolved.append(nurse)
    if problems:
        return {"error": "fix these names first, nothing was sent", "problems": problems}

    client = db.get_client()
    batch_id = str(uuid.uuid4())
    sent, skipped, failed = [], [], []
    seen = set()

    for nurse in resolved:
        if nurse["id"] in seen:
            continue
        seen.add(nurse["id"])

        # Don't double-text a carer who already has this exact offer open.
        open_offer = (
            client.table("sms_adhoc_offers").select("id")
            .eq("nurse_id", nurse["id"]).eq("date", date)
            .eq("shift_type", shift_type).eq("status", "offered")
            .limit(1).execute()
        )
        if open_offer.data:
            skipped.append(nurse["first_name"])
            continue

        msg = offer_message(nurse, fac, date, shift_type)
        row = client.table("sms_adhoc_offers").insert({
            "nurse_id": nurse["id"],
            "facility_id": fac["id"],
            "facility_name": fac["name"],
            "date": date,
            "shift_type": shift_type,
            "message": msg,
            "status": "offered",
            "offered_at": "now()",
            "batch_id": batch_id,
        }).execute()
        offer_id = row.data[0]["id"]
        try:
            db.send_sms(nurse["phone"], msg)
            sent.append(nurse["first_name"])
        except Exception:
            logger.exception("Failed to text %s", nurse["first_name"])
            db.mark_adhoc_offer(offer_id, "failed")
            failed.append(nurse["first_name"])

    return {
        "site": fac["name"], "date": date, "shift_type": shift_type,
        "texted": sent, "already_asked_and_waiting": skipped, "text_failed": failed,
    }


def tool_check_replies(date=None, **_):
    client = db.get_client()
    q = client.table("sms_adhoc_offers").select("*, nurses(first_name, last_name)")
    if date and _valid_date(date):
        q = q.eq("date", date)
    else:
        since = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        q = q.gte("created_at", since)
    rows = q.order("created_at", desc=True).limit(40).execute().data or []
    label = {"offered": "waiting", "accepted": "yes", "declined": "no", "failed": "text failed"}
    return {"offers": [{
        "carer": (r.get("nurses") or {}).get("first_name"),
        "site": r.get("facility_name"),
        "date": r["date"],
        "shift_type": r["shift_type"],
        "status": label.get(r["status"], r["status"]),
    } for r in rows]}



# --- create_shift ----------------------------------------------------------

BUBBLE_BASE = "https://knightingale.com.au/api/1.1/obj"

# Standard shift times per site: slug -> block -> [(code, start, end)].
SHIFT_TIMES = {
    "port_melbourne": {
        "Morning": [(None, "0700", "1500")],
        "Afternoon": [(None, "1430", "2215")],
        "Night": [(None, "2200", "0715")],
    },
    "mclean_lodge": {
        "Morning": [(None, "0730", "1530")],
        "Afternoon": [(None, "1515", "2230")],
        "Night": [(None, "2215", "0745")],
    },
    "williamstown": {
        "Morning": [("D6", "0600", "1400"), ("D7", "0700", "1500"), ("D9", "0900", "1700")],
        "Afternoon": [("A2", "1400", "2200"), ("A3", "1445", "2215")],
        "Night": [(None, "2200", "0700")],
    },
    "ron_con": {
        "Morning": [(None, "0700", "1500")],
        "Afternoon": [(None, "1400", "2200")],
        "Night": [(None, "2200", "0700")],
    },
    "angus_martin": {
        "Morning": [(None, "0700", "1500")],
        "Afternoon": [(None, "1500", "2200")],
        "Night": [(None, "2200", "0700")],
    },
    "eunice_seddon": {
        "Morning": [(None, "0700", "1500")],
        "Afternoon": [(None, "1430", "2215")],
        "Night": [(None, "2200", "0715")],
    },
    "gilgunya": {
        "Morning": [(None, "0700", "1500")],
        "Afternoon": [(None, "1430", "2100")],
        "Night": [(None, "2230", "0715")],
    },
    "brotherhood_st_laurence": {
        "Morning": [(None, "0700", "1500")],
        "Afternoon": [(None, "1445", "2215")],
        "Night": [(None, "2200", "0700")],
    },
}

EXTRA_LOCATION_IDS = {
    "gilgunya": "1782888571512x876198886010605600",
    "brotherhood_st_laurence": "1778383751696x989992016292812000",
}

ROLE_ALIASES = {
    "RN": "RN", "REGISTERED NURSE": "RN",
    "EN": "EN", "ENROLLED NURSE": "EN",
    "PCA": "PCA", "PERSONAL CARE ASSISTANT": "PCA", "PERSONAL CARE WORKER": "PCA",
    "PCW": "PCA", "AIN": "PCA",
    "DSW": "DSW", "DISABILITY SUPPORT WORKER": "DSW", "SUPPORT WORKER": "DSW",
}

MELB = ZoneInfo("Australia/Melbourne")


def _location_id(slug: str) -> str | None:
    if slug in EXTRA_LOCATION_IDS:
        return EXTRA_LOCATION_IDS[slug]
    for bubble_id, s in getattr(db, "LOCATION_ID_TO_SLUG", {}).items():
        if s == slug:
            return bubble_id
    return None


def _bubble_headers() -> dict:
    return {"Authorization": f"Bearer {os.environ['BUBBLE_API_TOKEN']}"}


# Victorian public holidays (from the `holidays` package data). Extend each year.
# Years not listed here are not checked and the reply says so.
VIC_PUBLIC_HOLIDAYS = {
    "2026-01-01", "2026-01-26", "2026-03-09", "2026-04-03", "2026-04-04",
    "2026-04-05", "2026-04-06", "2026-04-25", "2026-06-08", "2026-09-25",
    "2026-11-03", "2026-12-25", "2026-12-26", "2026-12-28",
    "2027-01-01", "2027-01-26", "2027-03-08", "2027-03-26", "2027-03-27",
    "2027-03-28", "2027-03-29", "2027-04-25", "2027-06-14", "2027-09-24",
    "2027-11-02", "2027-12-25", "2027-12-26", "2027-12-27", "2027-12-28",
}
HOLIDAY_YEARS = {2026, 2027}

# Residential Pricing record names in Bubble, by role then day kind.
PRICING_NAMES = {
    "EN": {"AM": "EN AM", "PM": "EN PM", "NS": "EN NT",
           "SAT": "EN SAT", "SUN": "EN SUN", "PH": "EN PH"},
    "PCA": {"AM": "PCA AM", "PM": "PCA PM", "NS": "PCA NT",
            "SAT": "PCA SAT", "SUN": "PCA SUN", "PH": "PCA P/H"},
    "RN": {"AM": "RN JNR AM", "PM": "RN JNR PM", "NS": "JNR RN NS",
           "SAT": "RN JNR SAT", "SUN": "RN JNR SUN", "PH": "RN JNR PH"},
}


def _day_kind(target, shift_type: str) -> tuple[str, bool]:
    """Pricing kind for a shift, by its START date: public holiday beats Sunday
    beats Saturday beats the weekday block. Returns (kind, holidays_checked)."""
    if target.year in HOLIDAY_YEARS:
        if target.isoformat() in VIC_PUBLIC_HOLIDAYS:
            return "PH", True
        checked = True
    else:
        checked = False
    wd = target.weekday()
    if wd == 6:
        return "SUN", checked
    if wd == 5:
        return "SAT", checked
    return {"Morning": "AM", "Afternoon": "PM", "Night": "NS"}[shift_type], checked


def _melb_date(iso: str):
    dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    return dt.astimezone(MELB).date()


def _get_pricing(name: str) -> dict | None:
    """Look up a Residential Pricing record in Bubble by its name. The Data API
    type is listed as "Residential Pricings", so that endpoint is tried first."""
    constraints = [{"key": "name", "constraint_type": "equals", "value": name}]
    last = None
    for endpoint in ("residentialpricings", "residentialpricing"):
        r = requests.get(
            f"{BUBBLE_BASE}/{endpoint}", headers=_bubble_headers(), timeout=20,
            params={"constraints": json.dumps(constraints), "limit": 5},
        )
        if r.status_code == 404:
            last = r
            continue
        r.raise_for_status()
        results = r.json().get("response", {}).get("results", [])
        return results[0] if results else None
    last.raise_for_status()
    return None


def _find_shift_template(location_id: str, start_num: int, role: str):
    """Most recent non-cancelled Bubble shift at this site with the same start time
    and role. Used ONLY for the things the pricing table does not hold: Bubble's
    own start/end numbers, the role value, address and supervisor."""
    constraints = [
        {"key": "location", "constraint_type": "equals", "value": location_id},
        {"key": "start time", "constraint_type": "equals", "value": start_num},
        {"key": "cancelled", "constraint_type": "equals", "value": "no"},
    ]
    r = requests.get(
        f"{BUBBLE_BASE}/shift", headers=_bubble_headers(), timeout=20,
        params={"constraints": json.dumps(constraints), "limit": 50,
                "sort_field": "Created Date", "descending": "true"},
    )
    r.raise_for_status()
    for t in r.json().get("response", {}).get("results", []):
        roles = {ROLE_ALIASES.get(str(x).strip().upper()) for x in (t.get("roles") or [])}
        if role in roles and t.get("end time") is not None:
            return t
    return None


def tool_create_shift(nurse_name, facility, date, shift_type, shift_code=None, **_):
    if db.DEV:
        return {"error": "KLARRA_MODE is dev, so no shift was created."}
    if not _valid_date(date) or shift_type not in BLOCKS:
        return {"error": "bad date or shift_type"}
    target = datetime.strptime(date, "%Y-%m-%d").date()
    if target < _now_melb().date():
        return {"error": "that date is in the past, nothing created"}

    nurse, err = _resolve_nurse(nurse_name)
    if err:
        return err
    fac = db.find_facility_by_name(facility)
    if not fac:
        return {"error": f"could not match a site called '{facility}'"}
    slug = fac["slug"]
    options = (SHIFT_TIMES.get(slug) or {}).get(shift_type)
    if not options:
        return {"error": f"no standard {shift_type} times saved for {fac['name']}"}

    if len(options) > 1:
        wanted = (shift_code or "").strip().upper()
        pick = next((o for o in options if o[0] == wanted), None)
        if not pick:
            return {"needs_choice": [f"{c} {a}-{b}" for c, a, b in options],
                    "site": fac["name"], "shift_type": shift_type}
    else:
        pick = options[0]
    _code, start, end = pick

    role = str(nurse.get("role") or "").upper()
    if role not in ROLES:
        return {"error": f"{nurse['first_name']} has no staffed role on file"}
    location_id = _location_id(slug)
    carer_bid = db.nurse_bubble_id(nurse["id"])
    if not location_id:
        return {"error": f"no Bubble location id saved for {fac['name']}"}
    if not carer_bid:
        return {"error": f"{nurse['first_name']} has no Bubble id, cannot create the shift"}

    clash = (db.get_client().table("shifts").select("id")
             .eq("nurse_id", nurse["id"]).eq("date", date)
             .neq("status", "cancelled").limit(1).execute().data)
    if clash:
        return {"error": f"{nurse['first_name']} already has a shift on {date}. Nothing created."}

    if role not in PRICING_NAMES:
        return {"error": (f"{role} shifts use NDIS pricing, not residential. "
                          "If this is an NDIS shift, ask Paul for the participant. "
                          "Nothing created.")}
    kind, holidays_checked = _day_kind(target, shift_type)
    pricing_name = PRICING_NAMES[role][kind]
    try:
        pricing = _get_pricing(pricing_name)
    except requests.HTTPError as e:
        return {"error": ("could not read the Residential Pricing table from Bubble "
                          f"({e.response.status_code}). Check it is ticked in Bubble "
                          "Settings, API, Data API. Nothing created.")}
    if not pricing:
        return {"error": f"no Residential Pricing record named '{pricing_name}'. Nothing created."}

    template = _find_shift_template(location_id, int(start), role)
    if not template:
        return {"error": (f"no earlier {role} {shift_type} shift at {fac['name']} "
                          f"({start}) to copy the role, address and supervisor from. "
                          "Create the first one in Bubble.")}

    rate = float(pricing.get("rate") or 0)
    carer_pay = float(pricing.get("carer pay") or 0)
    hourly_rev = float(pricing.get("hourly revenue") or 0)

    s_dt = datetime(target.year, target.month, target.day,
                    int(start[:2]), int(start[2:]), tzinfo=MELB)
    e_dt = datetime(target.year, target.month, target.day,
                    int(end[:2]), int(end[2:]), tzinfo=MELB)
    if e_dt <= s_dt:
        e_dt = (datetime(target.year, target.month, target.day) + timedelta(days=1)).replace(
            hour=int(end[:2]), minute=int(end[2:]), tzinfo=MELB)
    midnight = datetime(target.year, target.month, target.day, tzinfo=MELB)

    # Facility shifts have a 30 minute unpaid break: an 8h shift is 7.5h.
    hours = round((e_dt - s_dt).total_seconds() / 3600 - 0.5, 2)

    payload = {
        "accepted": "yes", "cancelled": "no", "attended": "no",
        "invoiced": "no", "csv": "no",
        "carer": carer_bid,
        "location": location_id,
        "date": midnight.isoformat(),
        "start time": template["start time"],
        "end time": template["end time"],
        "hours": hours,
        "rate": rate,
        "fee": round(hours * rate, 2),
        "wage": round(hours * carer_pay, 2),
        "revenue": round(hours * hourly_rev, 2),
        "roles": template.get("roles"),
        "address": template.get("address"),
        "res pricing": pricing["_id"],
        "supervisor": template.get("supervisor"),
        "check in time": s_dt.isoformat(),
        "check out time": e_dt.isoformat(),
    }
    payload = {k: v for k, v in payload.items() if v is not None}

    r = requests.post(f"{BUBBLE_BASE}/shift", headers=_bubble_headers(),
                      json=payload, timeout=20)
    if not r.ok:
        logger.error("Bubble rejected shift create: %s %s %s", r.status_code, r.text, payload)
        return {"error": f"Bubble rejected the shift ({r.status_code}). Nothing created."}
    new_id = r.json().get("id")
    if not new_id:
        return {"error": "Bubble gave no shift id back. Check Bubble before retrying."}

    problems = []
    try:
        db.upsert_shift_from_push(
            bubble_shift_id=new_id, nurse_id=nurse["id"], date=date,
            shift_type=shift_type, start_time=s_dt.isoformat(), end_time=e_dt.isoformat(),
            status="confirmed", facility_id=fac["id"],
        )
        db.assign_availability(nurse["id"], date, shift_type)
        avail_bid = db.get_availability_bubble_id(nurse["id"], date, shift_type)
        if avail_bid:
            ar = requests.patch(f"{BUBBLE_BASE}/availability/{avail_bid}",
                                headers=_bubble_headers(), json={"available": False},
                                timeout=15)
            ar.raise_for_status()
    except Exception:
        logger.exception("Shift %s created in Bubble but follow-up sync failed", new_id)
        problems.append("shift is in Bubble but the Supabase or availability update failed")

    code_txt = f"{_code} " if _code else ""
    label = BLOCK_LABEL[shift_type]
    text = (f"Created: {nurse['first_name']}, {fac['name']}, {db.short_date(date)}, "
            f"{label} {code_txt}{start}-{end} ({hours:g}h). Pricing: {pricing_name}.")
    if not holidays_checked:
        text += " Public holidays are not checked for that year, so check the rate."
    if problems:
        text += " Warning: " + "; ".join(problems) + "."
    return {"reply_text": text, "bubble_shift_id": new_id}



# --- create_ndis_shift -----------------------------------------------------

# NDIS Pricing records by item number: (prefix, day kind) -> "item no".
NDIS_ITEM_NOS = {
    ("01", "daytime"): "01_011_0107_1_1",
    ("01", "evening"): "01_015_0107_1_1",
    ("01", "night"): "01_002_0107_1_1",
    ("01", "saturday"): "01_013_0107_1_1",
    ("01", "sunday"): "01_014_0107_1_1",
    ("01", "public_holiday"): "01_012_0107_1_1",
    ("04", "daytime"): "04_104_0125_6_1",
    ("04", "evening"): "04_103_0125_6_1",
    ("04", "saturday"): "04_105_0125_6_1",
    ("04", "sunday"): "04_106_0125_6_1",
    ("04", "public_holiday"): "04_102_0125_6_1",
    # 04 night does not exist in the NDIS Pricing table.
}


def _parse_hhmm(t: str) -> tuple[int, int] | None:
    digits = "".join(c for c in str(t) if c.isdigit())
    if len(digits) == 3:
        digits = "0" + digits
    if len(digits) != 4:
        return None
    h, m = int(digits[:2]), int(digits[2:])
    return (h, m) if h < 24 and m < 60 else None


def _get_ndis_pricing(item_no: str) -> dict | None:
    constraints = [{"key": "item no", "constraint_type": "equals", "value": item_no}]
    last = None
    for endpoint in ("ndispricing", "ndispricings"):
        r = requests.get(
            f"{BUBBLE_BASE}/{endpoint}", headers=_bubble_headers(), timeout=20,
            params={"constraints": json.dumps(constraints), "limit": 5},
        )
        if r.status_code == 404:
            last = r
            continue
        r.raise_for_status()
        results = r.json().get("response", {}).get("results", [])
        return results[0] if results else None
    last.raise_for_status()
    return None


def _bubble_user(bubble_id: str) -> dict:
    r = requests.get(f"{BUBBLE_BASE}/user/{bubble_id}", headers=_bubble_headers(), timeout=20)
    r.raise_for_status()
    return r.json().get("response", {})


def _resolve_participant(name: str) -> tuple[dict | None, dict | None]:
    """Match a typed participant name against the participants table (first names).
    Returns (participant_row, None) or (None, error_dict)."""
    import difflib
    parts = name.strip().split()
    if not parts:
        return None, {"name": name, "problem": "not_found"}
    first, last = parts[0].lower(), (" ".join(parts[1:]).lower() or None)
    rows = db.get_client().table("participants").select("id, name, bubble_id").execute().data or []
    matches = [r for r in rows if (r.get("name") or "").strip().lower() == first]
    if not matches:
        names = {(r.get("name") or "").strip().lower(): r for r in rows}
        close = difflib.get_close_matches(first, names.keys(), n=3, cutoff=0.75)
        matches = [names[c] for c in close]
    if not matches:
        return None, {"name": name, "problem": "not_found"}
    if len(matches) == 1:
        return matches[0], None

    labelled = []
    for m in matches:
        try:
            u = _bubble_user(m["bubble_id"])
        except Exception:
            u = {}
        full = f"{m['name']} {u.get('last name') or ''}".strip()
        labelled.append((m, full))
    if last:
        narrowed = [m for m, full in labelled if full.lower().endswith(last)]
        if len(narrowed) == 1:
            return narrowed[0], None
    return None, {"name": name, "problem": "ambiguous", "matches": [f for _, f in labelled]}


def tool_create_ndis_shift(nurse_name, participant, date, start_time, end_time,
                           item_prefix, day_kind, **_):
    if db.DEV:
        return {"error": "KLARRA_MODE is dev, so no shift was created."}
    if not _valid_date(date):
        return {"error": "bad date"}
    target = datetime.strptime(date, "%Y-%m-%d").date()
    if target < _now_melb().date():
        return {"error": "that date is in the past, nothing created"}
    item_no = NDIS_ITEM_NOS.get((str(item_prefix), day_kind))
    if not item_no:
        return {"error": f"there is no NDIS item for {item_prefix} {day_kind}. Nothing created."}

    st, en = _parse_hhmm(start_time), _parse_hhmm(end_time)
    if not st or not en:
        return {"error": "could not read the start or end time"}
    if st[0] < 6:
        return {"error": "shifts starting between 00:00 and 05:59 are not supported yet. "
                         "Create that one in Bubble."}

    nurse, err = _resolve_nurse(nurse_name)
    if err:
        return err
    part, err = _resolve_participant(participant)
    if err:
        return err
    carer_bid = db.nurse_bubble_id(nurse["id"])
    if not carer_bid:
        return {"error": f"{nurse['first_name']} has no Bubble id, cannot create the shift"}

    s_dt = datetime(target.year, target.month, target.day, st[0], st[1], tzinfo=MELB)
    e_dt = datetime(target.year, target.month, target.day, en[0], en[1], tzinfo=MELB)
    overnight = e_dt <= s_dt
    if overnight:
        e_dt = (datetime(target.year, target.month, target.day) + timedelta(days=1)).replace(
            hour=en[0], minute=en[1], tzinfo=MELB)
    hours = round((e_dt - s_dt).total_seconds() / 3600, 2)
    if hours <= 0 or hours > 24:
        return {"error": "those times do not make a valid shift"}
    # Bubble stores time past midnight on an overnight shift as 24xx.
    start_num = st[0] * 100 + st[1]
    end_num = en[0] * 100 + en[1] + (2400 if overnight else 0)

    client = db.get_client()
    same_day = (client.table("shifts").select("start_time, end_time, participant_id")
                .eq("nurse_id", nurse["id"]).eq("date", date)
                .neq("status", "cancelled").execute().data or [])
    for row in same_day:
        try:
            rs = datetime.fromisoformat(str(row["start_time"]).replace("Z", "+00:00"))
            re_ = datetime.fromisoformat(str(row["end_time"]).replace("Z", "+00:00"))
        except Exception:
            return {"error": f"{nurse['first_name']} already has a shift on {date}. Nothing created."}
        if rs < e_dt and s_dt < re_:
            return {"error": f"{nurse['first_name']} already has a shift that overlaps. Nothing created."}
    dup = (client.table("shifts").select("id").eq("participant_id", part["id"])
           .eq("date", date).eq("start_time", s_dt.isoformat())
           .neq("status", "cancelled").limit(1).execute().data)
    if dup:
        return {"error": f"{part['name']} already has a shift at that time. Nothing created."}

    try:
        pricing = _get_ndis_pricing(item_no)
    except requests.HTTPError as e:
        return {"error": ("could not read the NDIS Pricing table from Bubble "
                          f"({e.response.status_code}). Nothing created.")}
    if not pricing:
        return {"error": f"no NDIS Pricing record with item no {item_no}. Nothing created."}

    try:
        pu = _bubble_user(part["bubble_id"])
    except Exception:
        return {"error": f"could not read {part['name']} from Bubble. Nothing created."}
    addr = pu.get("address")
    addr_text = addr.get("address") if isinstance(addr, dict) else addr

    price = float(pricing.get("item price") or 0)
    carer_pay = float(pricing.get("carer pay") or 0)
    hourly_rev = float(pricing.get("hourly revenue") or 0)
    midnight = datetime(target.year, target.month, target.day, tzinfo=MELB)

    payload = {
        "accepted": "yes", "cancelled": "no", "attended": "no", "invoiced": "no",
        "carer": carer_bid,
        "participant": part["bubble_id"],
        "coordinator": pu.get("coordinator"),
        "address": addr_text,
        "date": midnight.isoformat(),
        "start time": start_num,
        "end time": end_num,
        "hours": hours,
        "rate": price,
        "fee": round(hours * price, 2),
        "wage": round(hours * carer_pay, 2),
        "revenue": round(hours * hourly_rev, 2),
        "ndis pricing": pricing["_id"],
        "roles": ["DSW"],
        "recurring": "no",
    }
    payload = {k: v for k, v in payload.items() if v is not None}

    r = requests.post(f"{BUBBLE_BASE}/shift", headers=_bubble_headers(),
                      json=payload, timeout=20)
    if not r.ok:
        logger.error("Bubble rejected NDIS shift create: %s %s %s", r.status_code, r.text, payload)
        return {"error": f"Bubble rejected the shift ({r.status_code}). Nothing created."}
    new_id = r.json().get("id")
    if not new_id:
        return {"error": "Bubble gave no shift id back. Check Bubble before retrying."}

    shift_type = "Morning" if st[0] < 12 else "Afternoon" if st[0] < 18 else "Night"
    problems = []
    try:
        db.upsert_shift_from_push(
            bubble_shift_id=new_id, nurse_id=nurse["id"], date=date,
            shift_type=shift_type, start_time=s_dt.isoformat(), end_time=e_dt.isoformat(),
            status="confirmed", participant_id=part["id"],
        )
        db.assign_availability(nurse["id"], date, shift_type)
        avail_bid = db.get_availability_bubble_id(nurse["id"], date, shift_type)
        if avail_bid:
            ar = requests.patch(f"{BUBBLE_BASE}/availability/{avail_bid}",
                                headers=_bubble_headers(), json={"available": False},
                                timeout=15)
            ar.raise_for_status()
    except Exception:
        logger.exception("NDIS shift %s created in Bubble but follow-up sync failed", new_id)
        problems.append("shift is in Bubble but the Supabase or availability update failed")

    text = (f"Created NDIS shift: {nurse['first_name']} with {part['name']}, "
            f"{db.short_date(date)}, {st[0]:02d}{st[1]:02d}-{en[0]:02d}{en[1]:02d} "
            f"({hours:g}h). Pricing: {item_prefix} {day_kind.replace('_', ' ')}.")
    if problems:
        text += " Warning: " + "; ".join(problems) + "."
    return {"reply_text": text, "bubble_shift_id": new_id}


TOOL_FUNCS = {
    "check_availability": tool_check_availability,
    "check_nurse": tool_check_nurse,
    "offer_shift": tool_offer_shift,
    "check_replies": tool_check_replies,
    "create_shift": tool_create_shift,
    "create_ndis_shift": tool_create_ndis_shift,
}


def _call(name: str, args: dict):
    fn = TOOL_FUNCS.get(name)
    if not fn:
        return {"error": f"unknown tool {name}"}
    try:
        return fn(**(args or {}))
    except Exception as e:
        logger.exception("Tool %s failed", name)
        return {"error": f"{name} failed: {e}"}


# --- conversation memory ---------------------------------------------------

def _load_history(phone: str) -> list:
    try:
        r = (db.get_client().table("admin_threads").select("messages, updated_at")
             .eq("phone", phone).limit(1).execute())
    except Exception:
        logger.exception("Could not load admin thread")
        return []
    if not r.data:
        return []
    row = r.data[0]
    try:
        updated = datetime.fromisoformat(str(row["updated_at"]).replace("Z", "+00:00"))
        if datetime.now(timezone.utc) - updated > timedelta(minutes=THREAD_MINUTES):
            return []
    except Exception:
        return []
    msgs = row.get("messages") or []
    return [m for m in msgs if m.get("role") in ("user", "assistant") and m.get("content")]


def _save_history(phone: str, messages: list) -> None:
    messages = messages[-MAX_HISTORY:]
    while messages and messages[0]["role"] != "user":
        messages = messages[1:]
    try:
        db.get_client().table("admin_threads").upsert({
            "phone": phone,
            "messages": messages,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception:
        logger.exception("Could not save admin thread")


# --- entry point -----------------------------------------------------------

def run(phone: str, body: str) -> str | None:
    """Handle one text from Paul. Returns the reply text, or None if the message
    isn't for the admin agent (caller should continue normal routing)."""
    history = _load_history(phone)
    user_msg = {"role": "user", "content": body}
    messages = history + [user_msg]
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    for _ in range(MAX_STEPS):
        resp = client.messages.create(
            model=MODEL, max_tokens=700, system=_system(),
            tools=TOOLS, messages=messages,
        )
        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            if not text:
                return None
            _save_history(phone, history + [user_msg, {"role": "assistant", "content": text}])
            return text

        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            if block.name == "not_an_admin_request":
                return None
            logger.info("Admin agent tool %s %s", block.name, block.input)
            result = _call(block.name, block.input)
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result, default=str),
            })
        messages.append({"role": "user", "content": results})

    return "Sorry, that took too many steps. Can you try again?"


# --- group reply summary ---------------------------------------------------

def batch_summary(batch_id: str | None) -> str | None:
    """One-line summary once every carer in a group offer has replied. Returns None
    for single offers, missing batches, or while anyone is still outstanding."""
    if not batch_id:
        return None
    rows = (db.get_client().table("sms_adhoc_offers")
            .select("*, nurses(first_name)")
            .eq("batch_id", batch_id).order("created_at").execute().data or [])
    if len(rows) < 2 or any(r["status"] == "offered" for r in rows):
        return None
    parts = []
    for r in rows:
        name = (r.get("nurses") or {}).get("first_name") or "Carer"
        word = {"accepted": "YES", "declined": "NO"}.get(r["status"], "text failed")
        parts.append(f"{name} {word}")
    first = rows[0]
    return (f"All replied for {first.get('facility_name') or 'the site'} "
            f"{first['shift_type']} {db.short_date(first['date'])}: {', '.join(parts)}.")
