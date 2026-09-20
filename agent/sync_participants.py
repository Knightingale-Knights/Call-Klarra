"""
One-time sync: pull NDIS participants (Bubble account type = participant) into
Supabase's `participants` table, keyed on Bubble _id.

Run manually:  python agent/sync_participants.py
"""

import os
import logging
import requests

import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from dotenv import load_dotenv
import db

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("participant-sync")

BASE = "https://knightingale.com.au/api/1.1/obj"
TOKEN = os.environ["BUBBLE_API_TOKEN"]
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


def fetch_all(datatype: str, constraints: list | None = None) -> list[dict]:
    """Page through a Bubble data type, respecting its 100-per-call limit."""
    results, cursor = [], 0
    params = {"limit": 100}
    if constraints:
        import json
        params["constraints"] = json.dumps(constraints)
    while True:
        params["cursor"] = cursor
        r = requests.get(f"{BASE}/{datatype}", headers=HEADERS, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()["response"]
        batch = payload.get("results", [])
        results.extend(batch)
        if payload.get("remaining", 0) <= 0:
            break
        cursor += len(batch)
    return results


def upsert_participant(bubble_id: str, name: str) -> int:
    """Insert or update a participant keyed by their Bubble _id. Returns participant id."""
    client = db.get_client()
    existing = (
        client.table("participants").select("id")
        .eq("bubble_id", bubble_id).limit(1).execute()
    )
    if existing.data:
        pid = existing.data[0]["id"]
        client.table("participants").update({"name": name}).eq("id", pid).execute()
        return pid
    resp = client.table("participants").insert(
        {"name": name, "bubble_id": bubble_id}
    ).execute()
    return resp.data[0]["id"]


def sync_participants():
    users = fetch_all("user", constraints=[
        {"key": "account type", "constraint_type": "equals", "value": "participant"},
    ])
    logger.info("Fetched %d participants from Bubble", len(users))
    synced = 0
    skipped_no_name = 0
    for u in users:
        name = u.get("first name")
        if not name:
            skipped_no_name += 1
            continue
        upsert_participant(u["_id"], name)
        synced += 1
    logger.info("Synced %d participants (%d skipped, no name)", synced, skipped_no_name)


if __name__ == "__main__":
    sync_participants()
