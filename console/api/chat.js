// Klarra console chat endpoint.
// Claude answers questions about staff/shifts/facilities using read-only tools.
// Keys (ANTHROPIC_API_KEY, SUPABASE_*) stay server-side.

import Anthropic from "@anthropic-ai/sdk";
import { TOOLS, runTool } from "./_tools.js";

const MODEL = "claude-sonnet-4-6";

function melbourneToday() {
  // YYYY-MM-DD and a friendly form, in Australia/Melbourne.
  const now = new Date();
  const fmt = new Intl.DateTimeFormat("en-AU", {
    timeZone: "Australia/Melbourne",
    weekday: "long", year: "numeric", month: "long", day: "numeric",
  });
  const iso = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Australia/Melbourne",
    year: "numeric", month: "2-digit", day: "2-digit",
  }).format(now);
  return { iso, nice: fmt.format(now) };
}

function buildSystem() {
  const t = melbourneToday();
  return `You are Klarra, Knightingale's scheduling assistant, answering Paul's questions about his data.
Knightingale is a Melbourne aged-care / NDIS nursing staffing agency. Staff are called "carers", roles are EN and RN.
Today is ${t.nice} (${t.iso}), Melbourne time. Resolve "today", "tomorrow", "this week" from that. "Tomorrow" = the day after ${t.iso}.
Use the tools to look up real data before answering — never guess names, dates, or counts.
Always format any date as "Sunday, June 21" (weekday, month, day — no year).
Be concise and direct. Give the answer first; add detail only if useful. Plain text, minimal formatting.`;
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.status(405).json({ error: "POST only" });
    return;
  }
  try {
    const { messages } = req.body;
    if (!Array.isArray(messages)) {
      res.status(400).json({ error: "messages array required" });
      return;
    }

    const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
    const convo = [...messages];

    // Tool loop: keep letting Claude call tools until it returns a final answer.
    for (let i = 0; i < 6; i++) {
      const resp = await client.messages.create({
        model: MODEL,
        max_tokens: 1024,
        system: buildSystem(),
        tools: TOOLS,
        messages: convo,
      });

      const toolUses = resp.content.filter((b) => b.type === "tool_use");
      if (toolUses.length === 0) {
        const text = resp.content
          .filter((b) => b.type === "text")
          .map((b) => b.text)
          .join("\n");
        res.status(200).json({ reply: text });
        return;
      }

      convo.push({ role: "assistant", content: resp.content });
      const results = [];
      for (const tu of toolUses) {
        let out;
        try {
          out = await runTool(tu.name, tu.input || {});
        } catch (e) {
          out = { error: String(e.message || e) };
        }
        results.push({
          type: "tool_result",
          tool_use_id: tu.id,
          content: JSON.stringify(out).slice(0, 12000),
        });
      }
      convo.push({ role: "user", content: results });
    }

    res.status(200).json({ reply: "Sorry — that took too many steps. Try narrowing the question." });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: String(e.message || e) });
  }
}
