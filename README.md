# knightingale-agent

The Knightingale automated nurse-scheduling **voice agent**.

This is the long-running brain of the system: a LiveKit agent that answers/forwards
calls via Twilio, talks using OpenAI Realtime, and decides which nurse to call by
reading the decision policy in `skills/` and querying Supabase.

## What lives where

- **This repo (self-hosted, long-running):** the LiveKit voice agent + decision policy.
- **`claude` repo (Vercel, serverless):** stateless HTTP endpoints — SMS sending, webhooks.
- **Supabase:** data (facilities, nurses, approvals, availability, shifts), the
  `get_candidate_pool` function, and the `learned_decisions` store.

## Why the skill lives here

`skills/nurse-selection.md` is the policy the agent reads before every selection.
It lives in Git — not the database — so every change is a reviewable, version-controlled
commit. The agent **reads** this file; it never rewrites it. Learning happens in the
Supabase `learned_decisions` store (records of Paul's actual rulings), never by editing
policy. That separation is what keeps the system auditable.

## Structure

```
knightingale-agent/
  skills/
    nurse-selection.md      # the decision policy (read by the agent, edited by Paul)
  agent/                    # (Step 11) the LiveKit + OpenAI Realtime agent code
  README.md
```

## Status

- [x] Supabase data layer + get_candidate_pool function
- [x] Decision skill file (v0.1)
- [ ] learned_decisions store
- [ ] travel_km geocoding
- [ ] LiveKit + OpenAI Realtime agent
