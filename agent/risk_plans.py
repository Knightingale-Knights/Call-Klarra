"""
Risk Management Plans from questionnaire answers.

After a Risk Assessment or Home Safety questionnaire ends, every risk identified in that
participant's record gets a Bubble "Risk Management Plan", unless one already exists for it:
  risk               what the carer reported
  participant impact Low / Medium / High (rated by Claude)
  others             impact on carers: Low / Medium / High (rated by Claude)
  treatment          from the risk-emergency-generator API
  review date        12 months from creation
  person responsible "Paul"

What counts as a risk:
  Risk Assessment  every option ticked except "None" ("Other" uses the carer's words),
                   plus "other risks" unless it says none.
  Home Safety      any Yes on the hazard questions, any Unsafe item (with the carer's note).

process_pending() runs from the questionnaire worker loop.
"""

import os
import re
import json
import logging
from datetime import datetime, timedelta, timezone

import anthropic
import requests

import db
import questionnaire as q

logger = logging.getLogger("knightingale-agent.risk-plans")

MODEL = "claude-sonnet-4-6"
PLAN_TYPE = "riskmanagementplan"
RISK_API_URL = os.environ.get(
    "RISK_API_URL", "https://risk-emergency-generator.vercel.app/api/generate-risk-treatment")
SECTIONS = ("Risk Assessment", "Home Safety")
TEST_BASE_MARK = "/version-test/"

HAZARD_LABELS = {
    "smoker in home": "Smoking in the home",
    "infectious disease": "Infectious disease in the home",
    "pets": "Pets in the home",
    "hoarding": "Hoarding",
    "pests": "Pests or rodents in the home",
    "restraint used": "Use of restraint",
    "weapons": "Firearms or dangerous weapons in the home",
    "dangerous visitors": "Visitors who may pose a danger to carers",
    "faulty smoke detectors": "Missing or faulty smoke detectors",
    "doors locked": "Locked doors restricting movement",
    "power points near water": "Power points near water",
    "exposed gas cylinders": "Exposed gas cylinders",
}


def _none(v) -> bool:
    return str(v or "").strip().lower() in ("", "none", "n/a", "na", "no", "nil")


def identify_risks(section: str, record: dict) -> list[str]:
    """Base risk statements from a section record."""
    risks = []
    if section == "Risk Assessment":
        for f in q.SECTIONS[section]["fields"]:
            key = f["key"]
            if f["kind"] == "list":
                for opt in record.get(key) or []:
                    if opt == "None":
                        continue
                    if opt == "Other":
                        other = record.get(f"{key} other")
                        if not _none(other):
                            risks.append(f"{other} ({key})")
                    else:
                        risks.append(f"{opt} ({key})")
                # list without "Other" in its option set (e.g. Nutrition) may still have extras
                other = record.get(f"{key} other")
                if not _none(other) and "Other" not in (record.get(key) or []):
                    risks.append(f"{other} ({key})")
        if not _none(record.get("other risks")):
            risks.append(str(record["other risks"]).strip())
    elif section == "Home Safety":
        for f in q.SECTIONS[section]["fields"]:
            key, v = f["key"], str(record.get(f["key"]) or "")
            if f["kind"] == "answer" and v.lower() == "yes":
                risks.append(HAZARD_LABELS.get(key, key.capitalize()))
            elif f["kind"] == "safety" and v.lower() == "unsafe":
                risks.append(f"Home safety: {key}")
    return list(dict.fromkeys(risks))


def rate(section: str, risks: list[str], participant_name: str, notes: str) -> list[dict]:
    """Claude adds detail from the carer's notes and rates impact on participant and carers."""
    items = "\n".join(f"{i}. {r}" for i, r in enumerate(risks, 1))
    prompt = f"""These risks were identified for {participant_name}, an NDIS participant, from a carer's {section} answers.

Risks:
{items}

Carer's notes (may be empty):
{notes or "(none)"}

For each risk, in the same order, return ONLY a JSON array of objects:
[{{"detail": "", "participant_impact": "Low", "carer_impact": "Low"}}]
- detail: one short line from the carer's notes that describes this specific risk, or "" if the notes don't mention it. Never invent details.
- participant_impact / carer_impact: "Low", "Medium" or "High" for how much this risk could harm the participant, and the carers supporting them.
Never use an em dash."""
    try:
        msg = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]).messages.create(
            model=MODEL, max_tokens=2000, messages=[{"role": "user", "content": prompt}])
        m = re.search(r"\[.*\]", msg.content[0].text, re.S)
        out = json.loads(m.group(0)) if m else []
    except Exception:
        logger.exception("Risk rating failed")
        out = []
    fixed = []
    for i in range(len(risks)):
        r = out[i] if i < len(out) and isinstance(out[i], dict) else {}
        def lvl(v):
            v = str(v or "").strip().capitalize()
            return v if v in ("Low", "Medium", "High") else "Medium"
        fixed.append({"detail": str(r.get("detail") or "").strip().replace("—", ", "),
                      "participant_impact": lvl(r.get("participant_impact")),
                      "carer_impact": lvl(r.get("carer_impact"))})
    return fixed


def existing_risks(participant_bid: str) -> list[str]:
    rows = q.bubble_get(PLAN_TYPE, [{"key": "participant", "constraint_type": "equals",
                                     "value": participant_bid}])
    return [str(r.get("risk") or "").lower() for r in rows]


def treatment(participant_bid: str, risk: str) -> str:
    body = {"participantId": participant_bid, "risk": risk}
    if TEST_BASE_MARK in q.BUBBLE_BASE:
        body["version"] = "test"
    r = requests.post(RISK_API_URL, json=body, timeout=180)
    r.raise_for_status()
    return (r.json() or {}).get("riskTreatment", "").strip()


def create_plans(section: str, participant_bid: str, participant_name: str) -> int:
    """Create plans for new risks in this participant's section record. Returns count created."""
    record = q.get_record(section, participant_bid) or {}
    base_risks = identify_risks(section, record)
    have = existing_risks(participant_bid)
    new = [r for r in base_risks if not any(h.startswith(r.lower()) for h in have)]
    if not new:
        return 0
    ratings = rate(section, new, participant_name, record.get("home safety notes") or "")
    review = (datetime.now(timezone.utc) + timedelta(days=365)).isoformat()
    made = 0
    for base, rt in zip(new, ratings):
        risk = f"{base}: {rt['detail']}" if rt["detail"] else base
        try:
            text = treatment(participant_bid, risk)
        except Exception:
            logger.exception("Treatment API failed for %s", risk)
            text = ""
        try:
            q.bubble_create(PLAN_TYPE, {
                "participant": participant_bid,
                "risk": risk,
                "treatment": text,
                "participant impact": rt["participant_impact"],
                "others": rt["carer_impact"],
                "review date": review,
                "person responsible": "Paul",
            })
            made += 1
        except Exception:
            logger.exception("Could not create Risk Management Plan for %s", risk)
    logger.info("Created %d risk plans for %s (%s)", made, participant_name, section)
    return made


def process_pending() -> None:
    """Make plans for finished Risk Assessment / Home Safety sessions not yet processed."""
    since = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    r = (db.get_client().table("questionnaire_sessions")
         .select("id, section, participant_bubble_id, state")
         .in_("section", list(SECTIONS)).in_("status", ["Completed", "Partial"])
         .gte("created_at", since).execute())
    for s in r.data or []:
        state = s.get("state") or {}
        if not state or state.get("plans_done"):
            continue
        state["plans_done"] = True  # mark first: a failure never loops; re-runs dedupe anyway
        db.get_client().table("questionnaire_sessions").update(
            {"state": state}).eq("id", s["id"]).execute()
        default_base = q.BUBBLE_BASE
        q.BUBBLE_BASE = state.get("bubble_base") or default_base
        try:
            create_plans(s["section"], s["participant_bubble_id"],
                         state.get("participant_name") or "the participant")
        except Exception:
            logger.exception("Risk plans failed for session %s", s["id"])
        finally:
            q.BUBBLE_BASE = default_base
