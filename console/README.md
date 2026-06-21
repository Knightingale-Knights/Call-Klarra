# Klarra Console

Ask-anything web console over Knightingale's Supabase data. Read-only.

## Deploy (Vercel)
1. Push this folder to a new GitHub repo.
2. Vercel → New Project → import the repo.
3. Add Environment Variables:
   - `ANTHROPIC_API_KEY`
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_KEY`
4. Deploy. Open the URL.

## What it can read
nurses, facilities, availability, shifts, shift_requests — SELECT only.
No write path exists in the code.

## Notes
- Unlisted (noindex). Add auth before sharing the URL.
- Model: claude-sonnet-4-6.
- To add a table: add a tool in `api/_tools.js` (schema + a SELECT-only impl).
