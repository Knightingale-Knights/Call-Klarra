# Running the agent locally (Step 11)

Goal: hear the agent talk on your own machine before any phone wiring.

## 1. Get a LiveKit Cloud project (free)
1. Go to https://cloud.livekit.io and sign up.
2. Create a project.
3. Open **Settings → Keys** and copy: the **URL** (wss://...), **API Key**, **API Secret**.

## 2. Set up the code
In a terminal, from the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Add your keys
```bash
cp .env.example .env
```
Open `.env` and fill in:
- the three `LIVEKIT_` values from step 1,
- your `OPENAI_API_KEY`.
(Leave the Supabase ones for later.)

## 4. Talk to it
```bash
python agent/main.py console
```
`console` mode runs the agent right in your terminal using your mic and speakers —
no phone, no browser needed. You should hear it greet you. Say hello back and have a
short chat to confirm the voice loop works.

Press Ctrl+C to stop.

## What this proves
The speech-to-speech loop (your voice → OpenAI Realtime → its voice) runs end to end.
Once this works, the next steps add the decision tools (Supabase) and then the phone
layer (Twilio → LiveKit SIP).
