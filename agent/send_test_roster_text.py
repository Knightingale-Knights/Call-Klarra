"""
Sends ONE example weekly roster text to a phone, using the real message code from
recurring_shift_generator.py. Nothing is created or changed in Bubble or Supabase.

Send to your phone:  python3 agent/send_test_roster_text.py +61400000000
Or uses KLARRA_DEV_PHONE from .env if no number is given.
Preview only:        python3 agent/send_test_roster_text.py --print
"""

import os
import sys

import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from dotenv import load_dotenv

load_dotenv()

import db
import recurring_shift_generator as gen

args = [a for a in sys.argv[1:] if not a.startswith("--")]
preview = "--print" in sys.argv
phone = args[0] if args else os.environ.get("KLARRA_DEV_PHONE")
if not phone and not preview:
    sys.exit("Give a phone number, e.g. +61400000000, or set KLARRA_DEV_PHONE")
phone = phone or "preview"

FAKE_PARTICIPANTS = {1: "Jane Smith", 2: "Bob Lee"}
db.get_participant = lambda pid: {"id": pid, "name": FAKE_PARTICIPANTS[pid]}
db.get_nurse = lambda nid: {"id": nid, "first_name": "Sam", "phone": phone}
if preview:
    db.send_sms = lambda to, body: print(body)

lines = [
    gen.roster_line({"participant_id": 1, "start_time": "09:00", "end_time": "17:00"}, "2026-10-17"),
    gen.roster_line({"participant_id": 2, "start_time": "22:00", "end_time": "06:30"}, "2026-10-18"),
]
gen.send_roster_reminder(0, lines)
print("Done.")
