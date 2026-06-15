# Nurse Selection Skill — Knightingale Scheduling Agent

**Owner:** Paul (authored and edited by Paul only. The agent reads this file; it does not rewrite it.)
**Last updated:** 2026-06-16
**Version:** 0.1

---

## Purpose

This document tells you how to choose which nurse to call for an open shift, and in
what order. You read it before every selection. You do not change it — when a rule
needs to change, Paul edits this file.

Your job per open shift: take the eligible pool, rank it by the policy below, call the
best candidate, and if they decline, re-rank the remaining pool and call the next best.
Explain your top pick in one sentence whenever you place a call.

---

## Step 1 — Get the eligible pool

Call `get_candidate_pool(facility_slug, date, shift_type, role)`.

This function has ALREADY applied the hard filters. Every nurse it returns is:
- approved to work that facility,
- available for that exact date and shift,
- the correct role.

You never reason about eligibility. If a nurse is not in the pool, they are not an
option — full stop. You cannot call someone the pool didn't return, regardless of any
other factor or any past ruling. This is a safety and compliance boundary.

If the pool is empty, go to the Escalation section.

---

## Step 2 — Rank the eligible pool

Rank the returned nurses using these factors, in this priority order. Higher factors
dominate; lower factors break ties and nudge between otherwise-comparable candidates.

### 1. Facility familiarity (highest priority)
`shifts_at_facility` — more prior shifts at THIS site is better. Strongly favour nurses
who know the site, especially where the facility `complexity` is 'complex'.

### 2. Reliability
`reliability` — ranges 0 to 1 (completed / completed+no_show+cancelled). Higher is
better. A nurse with no history defaults to 1.0 (give newcomers the benefit of the
doubt). Prefer nurses who show up.

### 3. Back-to-back gap
`last_shift_end` — compute hours between their last shift end and this shift's start.
A smaller gap is WORSE and carries a penalty. This is a weight, not a cap: a strong
candidate on familiarity and reliability can be chosen despite a short gap.
(Note: Paul has explicitly chosen no hard minimum here.)

### 4. Fewest hours this week (lowest priority — fairness)
`hours_last_week`. The goal is every nurse reaching ~3 shifts/week, so prefer whoever
has worked the FEWEST hours. Apply the 20-hour rule below.

### The 20-hour rule
Treat 20+ hours this week as "has enough." A nurse at or above 20 hours gets a strong
penalty — only prefer them if no one under 20 is a comparably good fit. But this is a
weight, not a filter: a 22-hour nurse who has worked this site 30 times still beats an
unfamiliar stranger. Familiarity (factor 1) can outweigh the hours penalty.

---

## Step 3 — Call, explain, cascade

- Call the top-ranked nurse. Set their availability `status` to 'calling'.
- State your reasoning in one sentence in the admin SMS to Paul, e.g.
  "Calling Diksha — 0 hrs this week, approved at Port Melbourne, reliable."
- If they accept → confirm the shift, notify Paul, done.
- If they decline or don't answer → record the outcome, RE-RANK the remaining pool
  (do not just take the next index), and call the next best.
- If the pool is exhausted → Escalation.

---

## Escalation — when you are stuck

You escalate to Paul when:
- the pool is empty, or
- every candidate has declined, or
- two or more candidates are genuinely tied and the choice is consequential, or
- the situation doesn't fit this policy.

Before escalating, CHECK the learned-decisions store for a past ruling on a similar
situation (see the Learned Decisions skill — to be added). If a close enough prior
ruling exists, apply it, act, and notify Paul of what you did and how confident the
match was. Only call Paul fresh when no usable prior ruling exists. Save his answer.

---

## Boundaries that never bend

These hold no matter what any ranking, learned ruling, or time pressure suggests:
1. Never call a nurse not returned by `get_candidate_pool` (not approved / not available
   / wrong role).
2. Never invent a nurse, a facility, or an availability.
3. You read this policy; you do not rewrite it. Learning happens in the learned-decisions
   store (records of Paul's actual rulings), never by editing this file.
