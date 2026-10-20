"""
Emergency Plans: Klarra writes a participant's Bubble "Emergency Plan" on request.

Paul texts e.g. "create an emergency plan for Johanna" (admin_agent tool
create_emergency_plan). The plan is written in the background (it takes about a minute,
longer than Twilio waits for a reply), then Paul gets a text when it's saved.

Sources: the participant's User record, their questionnaire answers (Health Profile,
Daily Living, Risk Assessment, Home Safety), their Risk Management Plan risks and their
20 most recent progress notes.

Fields written (Bubble type "Emergency Plan", one per participant):
  fire evacuation / fire steps, hospital / hospital steps, infectious / infectious steps,
  natural disaster / natural disaster steps, power outage, backup power, supplies.
Only empty fields are filled, unless redo=True (then every field is rewritten).
"""

import os
import re
import json
import logging
import threading

import anthropic

import db
import questionnaire as q

logger = logging.getLogger("knightingale-agent.emergency-plans")

MODEL = "claude-sonnet-4-6"
PLAN_TYPE = "emergencyplan"
NOTE_TYPE = "progressnote"

FIELDS = {
    "fire evacuation": "Short summary (2 to 4 sentences) of the home's fire safety set-up for this "
                       "participant: exits, likely meeting point, smoke detectors, and anything about "
                       "their mobility, senses or behaviour that affects evacuating.",
    "fire steps": "Numbered step-by-step fire evacuation procedure for carers, specific to this "
                  "participant and home.",
    "hospital": "The nearest public hospital with an emergency department to the participant's "
                "address: name and street address. Include a phone number only if you are certain "
                "of it. End with: Please confirm this is the participant's preferred hospital.",
    "hospital steps": "When to call 000 versus taking the participant by car, then numbered steps for "
                      "getting them to hospital, including what to bring and what to tell staff "
                      "(conditions, medications, allergies).",
    "infectious": "Short summary of this participant's infection risks (conditions, medications, "
                  "living situation, infectious disease in the home) and why they matter.",
    "infectious steps": "Numbered steps for carers during an infectious disease outbreak or pandemic, "
                        "or if the participant or a carer becomes unwell.",
    "natural disaster": "Short summary of the natural hazards relevant to where the participant "
                        "lives (e.g. heatwave, storm, flood, bushfire) and what makes them vulnerable.",
    "natural disaster steps": "Numbered steps for carers to prepare for and respond to those hazards.",
    "power outage": "Numbered steps for carers in a power outage, covering anything this participant "
                    "relies on power for (equipment, lifts, lighting, heating or cooling, fridge-stored "
                    "medication).",
    "backup power": "What backup power and lighting the participant has or should have (torches, "
                    "batteries, power banks, equipment batteries), and how carers keep it ready.",
    "supplies": "Emergency supplies list tailored to this participant (medications, continence or "
                "personal care items, food and water for their diet, documents, mobility aids, "
                "phone charger, pet needs).",
}


def _profile_text(user: dict) -> str:
    keep = ["first name", "last name", "sex", "date of birth", "dob", "address", "about"]
    out = {}
    for k, v in user.items():
        if k.lower() in keep and v:
            out[k] = v.get("address") if isinstance(v, dict) and "address" in v else v
    return json.dumps(out, default=str, indent=1)


def _answers_text(pbid: str) -> str:
    parts = []
    for section in q.SECTION_ORDER:
        rec = q.get_record(section, pbid) or {}
        lines = []
        for f in q.SECTIONS[section]["fields"]:
            for key in (f["key"], f"{f['key']} other"):
                v = rec.get(key)
                if v in (None, "", []):
                    continue
                if isinstance(v, dict):
                    v = v.get("address") or v
                lines.append(f"- {key}: {', '.join(v) if isinstance(v, list) else v}")
        if lines:
            parts.append(f"{section}:\n" + "\n".join(lines))
    return "\n\n".join(parts) or "(no questionnaire answers yet)"


def _risks_text(pbid: str) -> str:
    try:
        rows = q.bubble_get("riskmanagementplan", [{"key": "participant",
                                                    "constraint_type": "equals", "value": pbid}])
        return "\n".join(f"- {r.get('risk')}" for r in rows if r.get("risk")) or "(none)"
    except Exception:
        return "(none)"


def _notes_text(pbid: str) -> str:
    try:
        import requests
        r = requests.get(f"{q.BUBBLE_BASE}/{NOTE_TYPE}", headers=q._headers(), timeout=30,
                         params={"constraints": json.dumps([{"key": "participant",
                                                             "constraint_type": "equals",
                                                             "value": pbid}]),
                                 "limit": 20, "sort_field": "Created Date", "descending": "true"})
        r.raise_for_status()
        rows = r.json()["response"].get("results", [])
    except Exception:
        logger.exception("Could not read progress notes")
        return "(none)"
    return "\n\n".join(f"Note {i} ({n.get('Created Date', '')[:10]}):\n{n.get('summary') or ''}"
                       for i, n in enumerate(rows, 1)) or "(none)"


def write_fields(user: dict, wanted: list[str]) -> dict:
    """Ask Claude for the wanted fields. Returns {field: text}."""
    pbid = user["_id"]
    p = q.first_name(user)
    pr = q.pronouns(user)
    spec = "\n".join(f'- "{k}": {FIELDS[k]}' for k in wanted)
    prompt = f"""You are writing parts of an NDIS Emergency Plan for {p}, a participant supported by Knightingale, a Melbourne aged care and NDIS agency. Refer to {p} by name or as {pr['subj']}/{pr['obj']}/{pr['poss']}.

<participant_profile>
{_profile_text(user)}
</participant_profile>

<questionnaire_answers>
{_answers_text(pbid)}
</questionnaire_answers>

<identified_risks>
{_risks_text(pbid)}
</identified_risks>

<recent_progress_notes>
{_notes_text(pbid)}
</recent_progress_notes>

Write these fields:
{spec}

Rules:
- Ground everything in {p}'s actual conditions, home, routines and recent events. Where information is missing, give sensible standard guidance; never invent facts about {p}.
- Never say information came from progress notes or records ("progress notes indicate", "documented", "recorded"). State facts directly.
- Contacts: only use names and numbers that appear above (emergency contacts, GP, pharmacy). Never invent phone numbers. 000 is fine.
- Refer to support staff as "carers", never "workers". Steps are written to the carer ("you").
- Plain text only: no markdown, no asterisks, no bullet symbols. Numbered steps as "1. ...". Never use em dashes or en dashes.
- Each field at most 250 words. Professional, NDIS Code of Conduct aligned, balancing duty of care with {p}'s dignity and choice.

Return ONLY a JSON object with exactly these keys: {json.dumps(wanted)}"""
    msg = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]).messages.create(
        model=MODEL, max_tokens=8000, messages=[{"role": "user", "content": prompt}])
    text = "".join(b.text for b in msg.content if b.type == "text")
    m = re.search(r"\{.*\}", text, re.S)
    data = json.loads(m.group(0)) if m else {}
    return {k: str(v).replace("—", ", ").replace("–", "-").strip()
            for k, v in data.items() if k in wanted and str(v or "").strip()}


def build(user: dict, redo: bool = False) -> tuple[int, str]:
    """Create or fill the participant's Emergency Plan. Returns (fields written, plan id)."""
    pbid = user["_id"]
    rows = q.bubble_get(PLAN_TYPE, [{"key": "participant", "constraint_type": "equals",
                                     "value": pbid}], limit=1)
    plan = rows[0] if rows else {"_id": q.bubble_create(PLAN_TYPE, {"participant": pbid})}
    wanted = [k for k in FIELDS if redo or not str(plan.get(k) or "").strip()]
    if not wanted:
        return 0, plan["_id"]
    fields = write_fields(user, wanted)
    if fields:
        q.bubble_patch(PLAN_TYPE, plan["_id"], fields)
    return len(fields), plan["_id"]


def build_in_background(user: dict, notify_phone: str, redo: bool = False) -> None:
    """Run build() on a thread and text Paul the result."""
    base = q.BUBBLE_BASE
    p = q.first_name(user)

    def run():
        q.BUBBLE_BASE = base
        try:
            n, _ = build(user, redo)
            msg = (f"{p}'s emergency plan is ready in Bubble ({n} sections written)."
                   if n else f"{p}'s emergency plan already has every section filled. "
                             f"Ask me to redo it if you want it rewritten.")
        except Exception as e:
            logger.exception("Emergency plan failed for %s", p)
            msg = f"Sorry, I couldn't finish {p}'s emergency plan ({e})."
        try:
            db.send_sms(notify_phone, msg)
        except Exception:
            logger.exception("Could not text emergency plan result")

    threading.Thread(target=run, daemon=True).start()
