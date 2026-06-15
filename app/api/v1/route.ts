// GET /api/v1 — discovery document. Lists endpoints, params and enums so the
// API is self-describing without leaving the response.

import { SITE, SORT_FIELDS, CAPABILITIES, json, apiHeaders } from "@/lib/api";

export const revalidate = 3600;

export function OPTIONS() {
  return new Response(null, { status: 204, headers: apiHeaders });
}

export function GET() {
  return json({
    name: "modelgrep API",
    version: "v1",
    description:
      "Free, read-only JSON API for the modelgrep LLM catalog — benchmarks, live speed/latency, pricing and capabilities for 300+ models on OpenRouter. No API key required.",
    docs: `${SITE}/api`,
    auth: "none",
    rate_limit: "fair use; responses are cached ~1h at the edge",
    endpoints: {
      "GET /api/v1/models": {
        description: "List & filter models.",
        query: {
          q: "search across id + name (space-separated terms, OR-matched)",
          maker: "filter by maker slug, e.g. anthropic, openai, google",
          provider: "filter by serving provider substring",
          free: "1 to return only $0 input-price models",
          benchmarked: "1 to return only models with benchmark data",
          capabilities: "comma list: " + CAPABILITIES.join(", "),
          max_price: "max input price, USD per million tokens",
          min_context: "min context window in tokens",
          sort: SORT_FIELDS.join(" | "),
          order: "asc | desc (defaults by field)",
          limit: "1–200 (default 50)",
          offset: "pagination offset (default 0)",
        },
      },
      "GET /api/v1/models/{id}": {
        description: "Single model with full benchmark detail + per-provider breakdown.",
        example: `${SITE}/api/v1/models/anthropic/claude-sonnet-4.5`,
      },
      "GET /api/v1/rankings/{collection}": {
        description:
          'Ranked "best LLM for X" list. Optionally scope to a maker: /rankings/{collection}/{maker}.',
        collections:
          "smartest, coding, design, fastest, lowest-latency, cheapest, free, reasoning, vision, agents, open-source, long-context, small",
        example: `${SITE}/api/v1/rankings/coding`,
      },
      "GET /api/v1/makers": {
        description: "Every model maker with model counts and its headline models.",
        example: `${SITE}/api/v1/makers`,
      },
    },
  });
}
