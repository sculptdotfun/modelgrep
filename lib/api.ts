// Public JSON API layer — a stable, serializable projection over the cached
// catalog. The site's own pages read Model directly; the API exposes a curated,
// versioned shape so we can evolve internals without breaking consumers.

import type { Model, ModelDetail } from "./types";
import type { Maker } from "./makers";
import { modelOwner } from "./format";

export const SITE = "https://modelgrep.com";

// CORS + CDN caching for a free, public, read-only API. The heavy work is
// already memoized in getCatalog() (hourly ISR); these headers let the edge and
// shared caches reuse responses too.
export const apiHeaders: Record<string, string> = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, OPTIONS",
  "Cache-Control": "public, s-maxage=3600, stale-while-revalidate=86400",
};

export function json(body: unknown, init?: { status?: number }): Response {
  return new Response(JSON.stringify(body, null, 2), {
    status: init?.status ?? 200,
    headers: { "Content-Type": "application/json; charset=utf-8", ...apiHeaders },
  });
}

export function apiError(status: number, message: string, extra?: Record<string, unknown>): Response {
  return json({ error: { status, message }, ...extra }, { status });
}

// ---- Serialization ----------------------------------------------------------
// One model → the public shape. Grouped (pricing/performance/benchmarks) so the
// payload self-documents and leaves room to add fields without flattening churn.

export function apiModel(m: Model) {
  return {
    id: m.id,
    name: m.name,
    maker: modelOwner(m.id),
    description: m.description || null,
    context_length: m.context_length || null,
    max_output: m.max_output,
    pricing: {
      input: m.price_input,
      output: m.price_output,
      cache_read: m.price_cache_read,
      caching: m.supports_caching,
      unit: "usd_per_million_tokens",
    },
    performance: {
      throughput_tps: m.throughput || null,
      latency_ms: m.latency,
      uptime: m.uptime,
    },
    capabilities: m.capabilities,
    modality: { input: m.input_modalities, output: m.output_modalities },
    providers: m.providers,
    knowledge_cutoff: m.knowledge_cutoff,
    hugging_face_id: m.hugging_face_id,
    is_moderated: m.is_moderated,
    created: m.created,
    benchmarks: {
      artificial_analysis: m.aa,
      design_arena: m.da,
    },
    url: `${SITE}/models/${m.id}`,
  };
}

export function apiModelDetail(m: ModelDetail) {
  return { ...apiModel(m), providers_detail: m.provider_details };
}

// One maker → its public summary, with the headline model in each axis.
export function apiMaker(mk: Maker) {
  return {
    slug: mk.slug,
    name: mk.displayName,
    model_count: mk.models.length,
    smartest: mk.bestIntel?.id ?? null,
    cheapest: mk.cheapest?.id ?? null,
    fastest: mk.fastest?.id ?? null,
    url: `${SITE}/makers/${mk.slug}`,
  };
}

// ---- Query (filter + sort + paginate) ---------------------------------------
// Mirrors the dashboard's selectModels() semantics so /api results match what
// the site shows for the same filters.

export type SortField =
  | "intelligence"
  | "coding"
  | "agentic"
  | "design"
  | "throughput"
  | "latency"
  | "price_input"
  | "price_output"
  | "context"
  | "created";

const SORT_VALUE: Record<SortField, (m: Model) => number | null> = {
  intelligence: (m) => m.aa?.intelligence ?? null,
  coding: (m) => m.aa?.coding ?? null,
  agentic: (m) => m.aa?.agentic ?? null,
  design: (m) => m.da?.elo ?? null,
  throughput: (m) => m.throughput || null,
  latency: (m) => m.latency,
  price_input: (m) => m.price_input,
  price_output: (m) => m.price_output,
  context: (m) => m.context_length || null,
  created: (m) => m.created || null,
};

// Lower-is-better fields default to ascending, like the dashboard.
const ASC_DEFAULT: SortField[] = ["latency", "price_input", "price_output"];

const CAP_KEYS = ["tools", "reasoning", "vision", "structured", "audio_in", "image_out"] as const;

export interface QueryParams {
  q?: string;
  provider?: string;
  maker?: string;
  free?: boolean;
  benchmarked?: boolean;
  caps?: string[]; // subset of CAP_KEYS
  max_price?: number | null; // $/M input
  min_context?: number | null; // tokens
  sort?: SortField;
  order?: "asc" | "desc";
  limit: number;
  offset: number;
}

export interface QueryResult {
  data: ReturnType<typeof apiModel>[];
  total: number; // matched before pagination
}

export function queryModels(catalog: Model[], p: QueryParams): QueryResult {
  const terms = (p.q ?? "").toLowerCase().split(/\s+/).filter(Boolean);
  const reqCaps = (p.caps ?? []).filter((c): c is (typeof CAP_KEYS)[number] =>
    (CAP_KEYS as readonly string[]).includes(c),
  );

  const matched = catalog.filter((m) => {
    if (terms.length) {
      const hay = `${m.id} ${m.name}`.toLowerCase();
      if (!terms.some((t) => hay.includes(t))) return false;
    }
    if (p.maker && modelOwner(m.id) !== p.maker.toLowerCase()) return false;
    if (p.provider && !m.providers.some((x) => x.toLowerCase().includes(p.provider!.toLowerCase())))
      return false;
    if (p.free && (m.price_input ?? 0) > 0) return false;
    if (p.benchmarked && !m.aa && !m.da) return false;
    if (p.max_price != null && (m.price_input ?? Infinity) > p.max_price) return false;
    if (p.min_context != null && (m.context_length ?? 0) < p.min_context) return false;
    for (const c of reqCaps) if (!m.capabilities[c]) return false;
    return true;
  });

  const field = p.sort ?? "intelligence";
  const dir = (p.order ?? (ASC_DEFAULT.includes(field) ? "asc" : "desc")) === "asc" ? 1 : -1;
  const val = SORT_VALUE[field];
  matched.sort((a, b) => {
    const av = val(a);
    const bv = val(b);
    if (av == null && bv == null) return 0;
    if (av == null) return 1; // nulls always sink
    if (bv == null) return -1;
    return (av - bv) * dir;
  });

  const page = matched.slice(p.offset, p.offset + p.limit);
  return { data: page.map(apiModel), total: matched.length };
}

// Parse + clamp query params off a request URL. Shared by the list endpoint.
export function parseQuery(url: URL): QueryParams {
  const g = (k: string) => url.searchParams.get(k);
  const num = (k: string): number | null => {
    const v = g(k);
    if (v == null || v === "") return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  };
  const bool = (k: string) => {
    const v = g(k);
    return v === "1" || v === "true";
  };
  const limit = Math.min(Math.max(num("limit") ?? 50, 1), 200);
  const offset = Math.max(num("offset") ?? 0, 0);
  const sort = (g("sort") ?? undefined) as SortField | undefined;
  const order = g("order") === "asc" ? "asc" : g("order") === "desc" ? "desc" : undefined;

  return {
    q: g("q") ?? undefined,
    provider: g("provider") ?? undefined,
    maker: g("maker") ?? undefined,
    free: bool("free"),
    benchmarked: bool("benchmarked"),
    caps: (g("capabilities") ?? "").split(",").map((s) => s.trim()).filter(Boolean),
    max_price: num("max_price"),
    min_context: num("min_context"),
    sort: sort && sort in SORT_VALUE ? sort : undefined,
    order,
    limit,
    offset,
  };
}

export const SORT_FIELDS = Object.keys(SORT_VALUE) as SortField[];
export const CAPABILITIES = CAP_KEYS;
