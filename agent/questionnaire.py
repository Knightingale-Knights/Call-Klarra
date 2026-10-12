"""
Participant questionnaires: field definitions plus Bubble + Supabase read/write.

Four sections, one Bubble record (and one Supabase row) per participant per section.
A field is "missing" when it is empty. Klarra only ever asks about missing fields.

Field kinds
  text    free text ("None" when there is nothing)
  answer  Answer option set: Yes / No / Unknown
  safety  Safety Check option set: Safe / Unsafe / N/A
  list    list of an option set; anything not in the set goes to "<field> other"
  notes   free text that is appended to, never replaced (home safety notes)

"<x> other" fields are never asked directly. They are filled by save_answer when a
list answer contains something not in the option set, and are only "missing" when
the list contains "Other".
"""

import os
import re
import json
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

import db

logger = logging.getLogger("knightingale-agent.questionnaire")

MELB = ZoneInfo("Australia/Melbourne")
BUBBLE_BASE = os.environ.get("QUESTIONNAIRE_BUBBLE_BASE", "https://knightingale.com.au/api/1.1/obj").rstrip("/")

ANSWER = ["Yes", "No", "Unknown"]
SAFETY = ["Safe", "Unsafe", "N/A"]

OPTIONS = {
    "Physical Health": ["None", "Allergies", "Arthritis", "Asthma", "Blood Disorders", "Bruising",
        "Cerebral Palsy", "Chronic Fatigue", "Cognitive Impairment", "Constipation", "COPD",
        "Decline Medical Exams", "Dementia", "Diabetes", "Dismiss Medical Advice", "Dysphasia",
        "Endocrine Conditions", "Epilepsy", "Fibromyalgia", "Fractures", "Heart Conditions",
        "Hearing Impairment", "Hepatitis", "Hypertension", "Incontinence", "Infectious Disease",
        "Missed Appointments", "Missing Medication", "Muscular Pain", "Nervous System Conditions",
        "Night Checks", "Obesity", "Seizures", "Respiratory Conditions", "Skin Conditions",
        "Sleep Apnea", "Sleep Disorders", "Smoking", "Stroke", "Teeth/Gum Conditions",
        "Visual Impairment", "Other"],
    "Mental Health": ["None", "ABI", "Auditory Hallucinations", "Depression", "Anxiety", "Bipolar",
        "Borderline Personality Disorder", "Delirium", "Hoarding", "Intellectual Impairment",
        "Mood Disorder", "Obsessive Compulsive Disorder", "Post-traumatic Stress Disorder",
        "Psychosis", "Schizophrenia", "Self Harm", "Self Neglect", "Suicide Risk", "Other"],
    "Personal Care": ["None", "Bathing", "Dental Hygiene", "Dressing", "Feeding", "Grooming",
        "Shaving", "Showering", "Skin Integrity", "Toileting", "Other"],
    "Nutrition": ["None", "Alertness", "Allergies", "Behaviours", "Choking", "Dehydration",
        "Enternal Feeds + Oral", "Enternal Feeds - Oral", "Environment", "Food Hiding/Hoarding",
        "Food Prep", "Food Refusal", "Modified Utensils", "Overnight Feeds", "Pica",
        "Posture/Positioning", "Seating Position", "Special Diet", "Swallowing Difficulty",
        "Textured Diet", "Thickened Fluids"],
    "Accidental Movement": ["None", "Bumping", "Falling", "Grabbing", "Holding", "Leaning",
        "Panicky", "Running", "Startle Reflex", "Sudden Body Movements", "Tripping", "Other"],
    "Manual Handling": ["None", "Mobility", "Moving In Bed", "Personal Care", "Transfers",
        "Vehicle Access", "Other"],
    "Environmental": ["None", "Absconding", "Electrocution", "Fire", "Poisons", "Sharps",
        "Smoking", "Sun", "Traffic", "Travel", "Water Hazard", "Other"],
    "Financial Risks": ["None", "Challenges Budgeting", "Debt", "Financially Exploitable",
        "Gambling", "Losing Wallet", "Low Income", "Managing Money", "Other"],
    "Social Risks": ["None", "Anti-Social Peers", "Criminal Behaviour", "Discrimination",
        "Domestic Violence", "Emotional Abuse", "Exploitation", "Family", "Harrassment",
        "Homelessness", "Lack of Informal Support", "Neglect", "Other Housemates",
        "Other Visitors", "Physical Abuse", "Property Damage", "Sexual Abuse", "Social Isolation",
        "Strangers", "Verbal Abuse", "Verbal Threats", "Weapons", "Other"],
    "Substance Use": ["None", "Alcohol", "Drugs", "Medication Misuse", "Smoking", "Other"],
}


def F(key, kind, topic, question, options=None, sensitive=False):
    return {"key": key, "kind": kind, "topic": topic, "question": question,
            "options": options, "sensitive": sensitive}


SECTIONS = {
    "Health Profile": {
        "type": "healthprofile", "table": "health_profile", "user_field": "health_profile",
        "fields": [
            F("emergency contact name one", "text", "emergency contacts",
              "Who is their main emergency contact? (name)"),
            F("emergency contact phone one", "text", "emergency contacts",
              "Main emergency contact's phone number."),
            F("emergency contact relationship one", "text", "emergency contacts",
              "Main emergency contact's relationship to them."),
            F("emergency contact name two", "text", "emergency contacts",
              "Is there a second emergency contact? Name, or 'None'."),
            F("emergency contact phone two", "text", "emergency contacts",
              "Second emergency contact's phone number ('None' if no second contact)."),
            F("emergency contact relationship two", "text", "emergency contacts",
              "Second emergency contact's relationship ('None' if no second contact)."),
            F("pharmacist name", "text", "their pharmacist",
              "Who is their pharmacist or pharmacy?"),
            F("pharmacist phone", "text", "their pharmacist",
              "Pharmacist's phone number."),
            F("ndis primary diagnosis", "text", "their NDIS diagnosis",
              "What is their primary formal NDIS diagnosis?"),
            F("ndis secondary diagnosis", "text", "their NDIS diagnosis",
              "Any secondary formal NDIS diagnosis? ('None' if not)"),
            F("physical health", "list", "physical health",
              "What physical health conditions do they have?", "Physical Health"),
            F("mental health", "list", "mental health",
              "What mental health conditions do they have?", "Mental Health", sensitive=True),
            F("medications", "text", "medications",
              "What medications do they take, for physical and mental health?"),
            F("allergies", "text", "allergies", "Do they have any allergies? ('None' if not)"),
            F("behaviour support plan", "answer", "their behaviour support plan",
              "Do they have a behaviour support plan?"),
            F("hospital admissions", "text", "hospital admissions",
              "Have they had any hospital admissions? When and why? ('None' if not)"),
            F("last hospital stay length", "text", "hospital admissions",
              "How long was their last hospital stay? ('N/A' if never admitted)"),
        ],
    },
    "Daily Living": {
        "type": "dailyliving", "table": "daily_living", "user_field": "daily_living",
        "fields": [
            F("living situation", "text", "their living situation",
              "What is their living situation? (who they live with, type of home)"),
            F("hobbies", "text", "hobbies", "What are their hobbies?"),
            F("foods liked", "text", "food likes and dislikes", "What foods do they like?"),
            F("foods disliked", "text", "food likes and dislikes", "What foods don't they like?"),
            F("dietary requirements", "text", "dietary requirements",
              "Do they have any dietary requirements? ('None' if not)"),
            F("mobility aids", "text", "mobility aids",
              "Do they use any mobility aids? Which ones? ('None' if not)"),
            F("daily tasks assistance", "text", "daily tasks",
              "What daily tasks do they need assistance with?"),
            F("personal care", "list", "personal care",
              "What personal care do they need assistance with?", "Personal Care"),
            F("supports needed", "text", "supports",
              "What kind of supports do they need?"),
            F("unmet needs", "text", "supports",
              "Is there any other assistance they need that they don't currently have?"),
            F("carer pref age", "text", "carer preferences",
              "Do they have a preferred carer age? ('No preference' is fine)"),
            F("carer pref gender", "text", "carer preferences",
              "Do they have a preferred carer gender?"),
            F("carer pref culture religion", "text", "carer preferences",
              "Any cultural, religious or ethnic preferences for their carer?"),
            F("carer pref language", "text", "carer preferences",
              "What language would they like their carer to speak?"),
            F("carer pref personality", "text", "carer preferences",
              "What personality type do they prefer in a carer?"),
            F("carer unique skills", "text", "carer skills and training",
              "Do carers need any unique skills or knowledge to support them?"),
            F("carer training required", "text", "carer skills and training",
              "Should carers undergo any specific training to support them?"),
            F("goals short term", "text", "their goals",
              "What goals would they like to achieve over the coming months?"),
            F("goals long term", "text", "their goals",
              "What goals would they like to achieve with our support over the coming years?"),
        ],
    },
    "Risk Assessment": {
        "type": "riskassessment", "table": "risk_assessment", "user_field": "risk_assessment",
        "fields": [
            F("nutrition", "list", "eating and drinking",
              "Are there any eating or drinking concerns?", "Nutrition"),
            F("accidental movement", "list", "accidental movements",
              "Do they have any accidental movements, like falling, tripping or grabbing?",
              "Accidental Movement"),
            F("manual handling", "list", "manual handling",
              "What manual handling assistance do they need?", "Manual Handling"),
            F("environmental", "list", "environmental risks",
              "Are there any environmental hazards or risks, like wandering off, fire, traffic or sharps?",
              "Environmental"),
            F("financial risks", "list", "financial risks",
              "Are there any concerns with how they manage money?", "Financial Risks",
              sensitive=True),
            F("social risks", "list", "social risks",
              "Are there any social risks, like isolation, difficult visitors or exploitation?",
              "Social Risks", sensitive=True),
            F("substance use", "list", "substance use",
              "Do they use alcohol, drugs, cigarettes, or misuse any medication?",
              "Substance Use", sensitive=True),
            F("other risks", "text", "other risks",
              "Are there any other risks we should be aware of? ('None' if not)"),
        ],
    },
    "Home Safety": {
        "type": "homesafety", "table": "home_safety", "user_field": "home_safety",
        "fields": [
            F("smoker in home", "answer", "home safety",
              "Does the participant or anyone they live with smoke?"),
            F("infectious disease", "answer", "home safety",
              "Is anyone in the home exposed to or suffering from an infectious disease?"),
            F("pets", "answer", "home safety", "Are there any pets?"),
            F("hoarding", "answer", "home safety", "Is there any evidence of hoarding?"),
            F("pests", "answer", "home safety", "Any evidence of pests or rodents?"),
            F("restraint used", "answer", "home safety", "Any evidence of restraint being used?"),
            F("weapons", "answer", "home safety", "Any firearms or dangerous weapons in the home?"),
            F("dangerous visitors", "answer", "home safety",
              "Any visitors or other people in the home who could pose a danger to carers?"),
            F("faulty smoke detectors", "answer", "home safety",
              "Are any smoke detectors missing or not working?"),
            F("doors locked", "answer", "home safety",
              "Are any doors kept locked so they can't move freely?"),
            F("power points near water", "answer", "home safety",
              "Are there power points near water?"),
            F("exposed gas cylinders", "answer", "home safety",
              "Any exposed gas cylinders that could be a danger?"),
            F("manual handling", "safety", "home safety", "Any manual handling issues in the home?"),
            F("walking aid access", "safety", "home safety",
              "Is access OK if using a walking aid?"),
            F("parking", "safety", "home safety", "Is parking safe?"),
            F("gates", "safety", "home safety", "Are the gates safe?"),
            F("pathway garden", "safety", "home safety", "Are the pathway and garden safe?"),
            F("stairs", "safety", "home safety",
              "Are the steps, stairs and railings safe? Not slippery or too narrow?"),
            F("verandah", "safety", "home safety",
              "Is the verandah or porch surface safe, including when wet?"),
            F("lighting", "safety", "home safety", "Is the lighting good? Do you feel safe?"),
            F("door access", "safety", "home safety",
              "Is the front door clear and easy to open?"),
            F("floor surfaces", "safety", "home safety", "Are the floor surfaces safe?"),
            F("power points", "safety", "home safety", "Are the power points in safe working order?"),
            F("electrical leads", "safety", "home safety",
              "Are electrical leads and extension cords safe, not a trip hazard?"),
            F("switches plugs", "safety", "home safety", "Are the switches and plugs safe?"),
            F("vacuum", "safety", "home safety", "Is the vacuum cleaner in good working order?"),
            F("bathroom access", "safety", "home safety",
              "Do they have good access to their bath, shower and toilet?"),
            F("bathroom drainage", "safety", "home safety",
              "Does the water drain well in the shower, bath and sinks?"),
            F("bathroom ventilation", "safety", "home safety", "Is the bathroom well ventilated?"),
            F("room to move", "safety", "home safety", "Is there enough space to move around?"),
            F("stove", "safety", "home safety", "Is the stove in working order? (N/A if none)"),
            F("fridge freezer", "safety", "home safety", "Is the fridge and freezer working?"),
            F("electrical equipment", "safety", "home safety",
              "Is any other electrical equipment causing an issue?"),
            F("food prep equipment", "safety", "home safety",
              "Is the food prep equipment safe? (N/A if none)"),
            F("tables chairs", "safety", "home safety", "Are the tables and chairs safe?"),
            F("washer dryer", "safety", "home safety",
              "Are the washing machine and dryer working OK?"),
            F("laundry drainage", "safety", "home safety", "Is the laundry drainage OK?"),
            F("laundry ventilation", "safety", "home safety", "Is the laundry ventilation OK?"),
            F("clothesline", "safety", "home safety",
              "Is the clothesline safe to use? (N/A if none)"),
            F("bed space", "safety", "home safety", "Is there enough space around the bed?"),
            F("bed height", "safety", "home safety", "Is the bed a suitable height?"),
            F("portable heaters", "safety", "home safety",
              "Are any portable heaters safe and working? (N/A if none)"),
            F("furniture placement", "safety", "home safety",
              "Does the furniture placement let them move around OK?"),
            F("cleaning products", "safety", "home safety",
              "Are the cleaning products in the home safe, not a hazard?"),
            F("cleaning products storage", "safety", "home safety",
              "Are cleaning products stored in a safe place?"),
            F("home safety notes", "notes", "home safety", ""),
        ],
    },
}

SECTION_ORDER = ["Health Profile", "Daily Living", "Risk Assessment", "Home Safety"]


# ---------- helpers ----------

def _headers():
    return {"Authorization": f"Bearer {os.environ['BUBBLE_API_TOKEN']}"}


def col(key: str) -> str:
    """Bubble field name -> Supabase column name."""
    return key.replace(" ", "_")


def field_def(section: str, key: str) -> dict | None:
    for f in SECTIONS[section]["fields"]:
        if f["key"] == key:
            return f
    return None


def _empty(v) -> bool:
    return v is None or (isinstance(v, str) and not v.strip()) or (isinstance(v, list) and not v)


def bubble_get(datatype: str, constraints: list, limit: int = 100, **params) -> list[dict]:
    results, cursor = [], 0
    while True:
        p = {"constraints": json.dumps(constraints), "limit": limit, "cursor": cursor, **params}
        r = requests.get(f"{BUBBLE_BASE}/{datatype}", headers=_headers(), params=p, timeout=30)
        r.raise_for_status()
        resp = r.json()["response"]
        batch = resp.get("results", [])
        results.extend(batch)
        if resp.get("remaining", 0) <= 0 or not batch:
            return results
        cursor += len(batch)


def bubble_get_by_id(datatype: str, bubble_id: str) -> dict | None:
    r = requests.get(f"{BUBBLE_BASE}/{datatype}/{bubble_id}", headers=_headers(), timeout=20)
    if r.status_code in (400, 404):
        return None
    r.raise_for_status()
    return r.json()["response"]


def bubble_user(bubble_id: str) -> dict:
    r = requests.get(f"{BUBBLE_BASE}/user/{bubble_id}", headers=_headers(), timeout=20)
    r.raise_for_status()
    return r.json()["response"]


_FIELD_ERR = re.compile(r"Invalid data for field ([^:]+): could not parse")


def _send(method: str, url: str, fields: dict) -> requests.Response:
    """Send to Bubble. If an option set field is rejected (option sets may be named in
    lowercase, e.g. 'health profile'), retry that field lowercased."""
    fields = dict(fields)
    for _ in range(len(fields) + 1):
        r = requests.request(method, url, headers=_headers(), json=fields, timeout=20)
        if r.ok:
            return r
        m = _FIELD_ERR.search(r.text)
        key = m.group(1).strip() if m else None
        if not key or key not in fields:
            break
        v = fields[key]
        low = [x.lower() for x in v] if isinstance(v, list) else str(v).lower()
        if low == v:
            break
        fields[key] = low
    logger.error("Bubble %s %s failed: %s %s", method, url, r.status_code, r.text)
    r.raise_for_status()
    return r


def bubble_patch(datatype: str, bubble_id: str, fields: dict) -> None:
    _send("PATCH", f"{BUBBLE_BASE}/{datatype}/{bubble_id}", fields)


def bubble_create(datatype: str, fields: dict) -> str:
    return _send("POST", f"{BUBBLE_BASE}/{datatype}", fields).json()["id"]


def first_name(user: dict) -> str:
    return (user.get("first name") or "").strip() or "there"


# ---------- section records ----------

def get_record(section: str, participant_bid: str) -> dict | None:
    rows = bubble_get(SECTIONS[section]["type"],
                      [{"key": "participant", "constraint_type": "equals",
                        "value": participant_bid}], limit=1)
    return rows[0] if rows else None


def get_or_create_record(section: str, participant_bid: str) -> dict:
    rec = get_record(section, participant_bid)
    if rec:
        return rec
    s = SECTIONS[section]
    new_id = bubble_create(s["type"], {"participant": participant_bid})
    try:  # link it from the participant's User record (best effort)
        bubble_patch("user", participant_bid, {s["user_field"]: new_id})
    except Exception:
        logger.warning("Could not set User.%s for %s", s["user_field"], participant_bid)
    return {"_id": new_id, "participant": participant_bid}


def missing_fields(section: str, record: dict | None) -> list[str]:
    """Askable empty fields, in order. '<x> other' and notes are never asked directly."""
    record = record or {}
    out = []
    for f in SECTIONS[section]["fields"]:
        if f["kind"] == "notes":
            continue
        if _empty(record.get(f["key"])):
            out.append(f["key"])
    return out


def recently_skipped(participant_bid: str, section: str, days: int = 14) -> set:
    """Fields skipped in this section's sessions over the last `days`, so a field
    nobody knows doesn't block the section forever."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    r = (db.get_client().table("questionnaire_sessions").select("skipped_fields")
         .eq("participant_bubble_id", participant_bid).eq("section", section)
         .gte("created_at", since).execute())
    out = set()
    for row in r.data or []:
        for k in (row.get("skipped_fields") or "").split(","):
            if k.strip():
                out.add(k.strip())
    return out


def next_section(participant_bid: str) -> tuple[str, list[str]] | None:
    """First section (in order) with askable missing fields."""
    for section in SECTION_ORDER:
        missing = missing_fields(section, get_record(section, participant_bid))
        skipped = recently_skipped(participant_bid, section)
        askable = [k for k in missing if k not in skipped]
        if askable:
            return section, askable
    return None


def topics_phrase(section: str, keys: list[str], limit: int = 3) -> str:
    topics = []
    for k in keys:
        t = field_def(section, k)["topic"]
        if t not in topics:
            topics.append(t)
    shown = topics[:limit]
    if len(topics) > limit:
        shown.append("a few other things")
    if len(shown) == 1:
        return shown[0]
    return ", ".join(shown[:-1]) + " and " + shown[-1]


# ---------- answers ----------

def _match_option(value: str, options: list[str]) -> str | None:
    v = value.strip().lower()
    for o in options:
        if o.lower() == v:
            return o
    for o in options:  # loose: "post traumatic stress" ~ "Post-traumatic Stress Disorder"
        ol = o.lower().replace("-", " ").replace("/", " ")
        vl = v.replace("-", " ").replace("/", " ")
        if len(vl) >= 3 and (vl in ol or ol in vl):
            return o
    return None


def normalise_answer(section: str, key: str, value: str) -> tuple[dict, str]:
    """Turn Klarra's spoken-answer value into Bubble field values.
    Returns (fields_to_write, message_back_to_klarra). Raises ValueError if invalid."""
    f = field_def(section, key)
    if not f:
        raise ValueError(f"'{key}' is not a field in {section}.")
    kind = f["kind"]
    value = (value or "").strip()
    if not value:
        raise ValueError("Empty value. Use skip_question if they don't know.")
    if kind in ("text", "notes"):
        return {key: value}, "Saved."
    if kind in ("answer", "safety"):
        opts = ANSWER if kind == "answer" else SAFETY
        m = _match_option(value, opts)
        if not m:
            raise ValueError(f"Value must be one of {opts}.")
        return {key: m}, "Saved."
    # list
    opts = OPTIONS[f["options"]]
    matched, extra = [], []
    for part in [p for p in value.replace(";", ",").split(",") if p.strip()]:
        m = _match_option(part, opts)
        if m and m not in matched:
            matched.append(m)
        elif not m:
            extra.append(part.strip())
    if "None" in matched and len(matched) > 1:
        matched.remove("None")
    out = {}
    if extra:
        if "Other" in opts and "Other" not in matched:
            matched.append("Other")
        other_key = f"{key} other"
        out[other_key] = ", ".join(extra)
    out[key] = matched
    msg = f"Saved as {matched}"
    if extra:
        msg += f", with '{', '.join(extra)}' recorded as other"
    return out, msg + "."


def save_fields(section: str, record_id: str, participant_bid: str, fields: dict,
                append_notes: bool = False) -> None:
    """Write to Bubble, then mirror to Supabase."""
    s = SECTIONS[section]
    if append_notes and "home safety notes" in fields:
        rec = get_record(section, participant_bid) or {}
        prev = (rec.get("home safety notes") or "").strip()
        fields = dict(fields)
        fields["home safety notes"] = (prev + "\n" if prev else "") + fields["home safety notes"]
    bubble_patch(s["type"], record_id, fields)
    try:
        row = {col(k): v for k, v in fields.items()}
        row.update({"participant_bubble_id": participant_bid, "bubble_id": record_id,
                    "updated_at": datetime.now(timezone.utc).isoformat()})
        db.get_client().table(s["table"]).upsert(row, on_conflict="participant_bubble_id").execute()
    except Exception:
        logger.exception("Supabase mirror failed for %s %s", section, list(fields))


# ---------- sessions ----------

def update_session(session: dict, **fields) -> None:
    """Update a session in Supabase (snake_case keys) and mirror to Bubble."""
    db.get_client().table("questionnaire_sessions").update(fields).eq("id", session["id"]).execute()
    bid = session.get("bubble_id")
    if not bid:
        return
    to_bubble = {k.replace("_", " "): v for k, v in fields.items()
                 if k in ("status", "started", "completed", "call_sid", "transcript",
                          "skipped_fields", "heads_up_sent", "scheduled_call")}
    if "carer_bubble_id" in fields:
        to_bubble["carer"] = fields["carer_bubble_id"]
    if to_bubble:
        try:
            bubble_patch("questionnairesession", bid, to_bubble)
        except Exception:
            logger.exception("Bubble session mirror failed for %s", bid)


def get_session(session_id: str) -> dict | None:
    r = db.get_client().table("questionnaire_sessions").select("*").eq("id", session_id).limit(1).execute()
    return r.data[0] if r.data else None


def melb_now() -> datetime:
    return datetime.now(MELB)


def shift_start(shift: dict) -> datetime | None:
    """Bubble shift date (Melbourne midnight) + start time number (e.g. 930)."""
    if not shift.get("date") or shift.get("start time") is None:
        return None
    d = datetime.fromisoformat(str(shift["date"]).replace("Z", "+00:00")).astimezone(MELB).date()
    n = int(shift["start time"])
    return datetime(d.year, d.month, d.day, n // 100, n % 100, tzinfo=MELB)


def pretty_time(dt: datetime) -> str:
    return dt.strftime("%I:%M%p").lstrip("0").lower().replace(":00", "")
