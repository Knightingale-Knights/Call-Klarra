// Read-only Supabase query helpers + tool schemas for the Klarra console.
// Every function here is SELECT-only. No insert/update/delete anywhere.

import { createClient } from "@supabase/supabase-js";

const sb = () =>
  createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_KEY);

// --- tool schemas given to Claude ---
export const TOOLS = [
  {
    name: "list_nurses",
    description:
      "List nurses (carers). Optionally filter by role (EN or RN) or by a name fragment. Returns name, role, phone, address.",
    input_schema: {
      type: "object",
      properties: {
        role: { type: "string", enum: ["EN", "RN"] },
        name: { type: "string", description: "case-insensitive name fragment" },
      },
    },
  },
  {
    name: "list_facilities",
    description: "List all facilities with their slug and complexity.",
    input_schema: { type: "object", properties: {} },
  },
  {
    name: "availability",
    description:
      "Find nurse availability. Filter by date (YYYY-MM-DD), shift_type (Morning/Afternoon/Night), and/or facility slug. Returns nurse name + the availability rows.",
    input_schema: {
      type: "object",
      properties: {
        date: { type: "string" },
        shift_type: { type: "string", enum: ["Morning", "Afternoon", "Night"] },
        facility_slug: { type: "string" },
      },
    },
  },
  {
    name: "shifts",
    description:
      "Look up worked/booked shifts. Filter by nurse name fragment, facility slug, and/or a date range (date_from, date_to as YYYY-MM-DD). Returns shift rows with nurse + facility names.",
    input_schema: {
      type: "object",
      properties: {
        nurse: { type: "string" },
        facility_slug: { type: "string" },
        date_from: { type: "string" },
        date_to: { type: "string" },
      },
    },
  },
  {
    name: "shift_requests",
    description:
      "List recent shift requests (the intake queue). Optionally filter by status (pending/working/filled/unfilled). Returns date, shift, role, status, facility.",
    input_schema: {
      type: "object",
      properties: { status: { type: "string" } },
    },
  },
];

// --- implementations ---
export async function runTool(name, input) {
  const db = sb();
  switch (name) {
    case "list_nurses": {
      let q = db.from("nurses").select("first_name,last_name,role,phone,address");
      if (input.role) q = q.eq("role", input.role);
      if (input.name) q = q.or(`first_name.ilike.%${input.name}%,last_name.ilike.%${input.name}%`);
      const { data, error } = await q.limit(200);
      if (error) throw error;
      return data;
    }
    case "list_facilities": {
      const { data, error } = await db
        .from("facilities")
        .select("name,slug,complexity");
      if (error) throw error;
      return data;
    }
    case "availability": {
      let q = db
        .from("availability")
        .select("date,shift_type,status,nurses(first_name,last_name),facilities(slug,name)");
      if (input.date) q = q.eq("date", input.date);
      if (input.shift_type) q = q.eq("shift_type", input.shift_type);
      const { data, error } = await q.limit(300);
      if (error) throw error;
      let rows = data || [];
      if (input.facility_slug)
        rows = rows.filter((r) => r.facilities?.slug === input.facility_slug);
      return rows;
    }
    case "shifts": {
      let q = db
        .from("shifts")
        .select("date,shift_type,start_time,end_time,status,nurses(first_name,last_name),facilities(slug,name)");
      if (input.date_from) q = q.gte("date", input.date_from);
      if (input.date_to) q = q.lte("date", input.date_to);
      const { data, error } = await q.order("date", { ascending: false }).limit(300);
      if (error) throw error;
      let rows = data || [];
      if (input.facility_slug)
        rows = rows.filter((r) => r.facilities?.slug === input.facility_slug);
      if (input.nurse) {
        const n = input.nurse.toLowerCase();
        rows = rows.filter((r) => {
          const full = `${r.nurses?.first_name || ""} ${r.nurses?.last_name || ""}`.toLowerCase();
          return full.includes(n);
        });
      }
      return rows;
    }
    case "shift_requests": {
      let q = db
        .from("shift_requests")
        .select("date,shift_type,role,status,source,facilities(name,slug)")
        .order("created_at", { ascending: false });
      if (input.status) q = q.eq("status", input.status);
      const { data, error } = await q.limit(100);
      if (error) throw error;
      return data;
    }
    default:
      throw new Error(`Unknown tool: ${name}`);
  }
}
