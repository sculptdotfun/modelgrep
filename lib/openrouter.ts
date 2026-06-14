// OpenRouter data client — ports the validated Python pipeline to TS.
//
// Sources (all keyless):
//   /api/v1/models                              base metadata, pricing, caps, cutoff
//   /api/v1/models/{id}/endpoints               providers, uptime, caching, quant
//   /<id>/performance (RSC)                     live p50 throughput/latency
//   /api/internal/v1/artificial-analysis-benchmarks?slug=<canonical>
//   /api/internal/v1/design-arena-benchmarks?slug=<canonical>

import type {
  AABenchmark,
  Capabilities,
  DABenchmark,
  Model,
  ProviderDetail,
} from "./types";

const BASE = "https://openrouter.ai/api/v1";
const INTERNAL = "https://openrouter.ai/api/internal/v1";
const SITE = "https://openrouter.ai";

const HEADERS: Record<string, string> = {
  "User-Agent":
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36",
  Referer: "https://openrouter.ai/",
  Accept: "application/json, text/plain, */*",
};

// Revalidate windows (seconds): perf/providers refresh hourly, benchmarks daily.
const REVAL_LIVE = 3600;
const REVAL_BENCH = 86400;

async function getJSON<T>(
  url: string,
  revalidate: number,
  fallback: T,
  retries = 2,
): Promise<T> {
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const res = await fetch(url, { headers: HEADERS, next: { revalidate } });
      if (res.ok) return (await res.json()) as T;
      // 4xx other than 429 won't recover on retry; bail early. 429/5xx back off.
      if (res.status !== 429 && res.status < 500) return fallback;
    } catch {
      // network/abort — retry
    }
    if (attempt < retries) await new Promise((r) => setTimeout(r, 400 * (attempt + 1)));
  }
  return fallback;
}

// ---- pricing helpers ---------------------------------------------------------

function perMillion(v: unknown): number | null {
  const n = typeof v === "string" ? parseFloat(v) : (v as number);
  if (v === null || v === undefined || v === "" || Number.isNaN(n)) return null;
  return Math.round(n * 1_000_000 * 10000) / 10000;
}

// ---- capabilities ------------------------------------------------------------

interface RawModel {
  id: string;
  canonical_slug?: string;
  name?: string;
  description?: string;
  context_length?: number;
  created?: number;
  knowledge_cutoff?: string | null;
  hugging_face_id?: string | null;
  pricing?: Record<string, string>;
  architecture?: {
    modality?: string;
    input_modalities?: string[];
    output_modalities?: string[];
  };
  top_provider?: {
    max_completion_tokens?: number | null;
    is_moderated?: boolean | null;
  };
  supported_parameters?: string[];
}

function capabilities(m: RawModel): Capabilities {
  const params = new Set(m.supported_parameters ?? []);
  const inputs = new Set(m.architecture?.input_modalities ?? ["text"]);
  const outputs = new Set(m.architecture?.output_modalities ?? ["text"]);
  return {
    tools: params.has("tools") || params.has("tool_choice"),
    reasoning: params.has("reasoning") || params.has("include_reasoning"),
    structured: params.has("structured_outputs") || params.has("response_format"),
    vision: inputs.has("image"),
    audio_in: inputs.has("audio"),
    image_out: outputs.has("image"),
  };
}

// ---- live performance (RSC scrape) ------------------------------------------

const PERF_RE = /"p50_throughput":([0-9.]+),"p50_latency":([0-9.]+)/g;

export async function fetchPerf(
  modelId: string,
): Promise<{ throughput: number; latency: number | null } | null> {
  try {
    const res = await fetch(`${SITE}/${modelId}/performance`, {
      headers: { ...HEADERS, RSC: "1" },
      next: { revalidate: REVAL_LIVE },
    });
    if (!res.ok) return null;
    const text = await res.text();
    let bestTp = 0;
    let bestLat = Infinity;
    for (const m of text.matchAll(PERF_RE)) {
      const tp = parseFloat(m[1]);
      const lat = parseFloat(m[2]);
      if (tp > bestTp) bestTp = tp;
      if (lat > 0 && lat < bestLat) bestLat = lat;
    }
    if (bestTp === 0 && bestLat === Infinity) return null;
    return {
      throughput: bestTp > 0 ? Math.round(bestTp * 10) / 10 : 0,
      latency: bestLat !== Infinity ? Math.round(bestLat) : null,
    };
  } catch {
    return null;
  }
}

// ---- benchmarks --------------------------------------------------------------

interface AARecord {
  benchmark_data?: { evaluations?: Record<string, number | null> };
  percentiles?: Record<string, number | null>;
}

export async function fetchAA(canonicalSlug: string): Promise<AABenchmark | null> {
  const data = await getJSON<{ data?: AARecord[] }>(
    `${INTERNAL}/artificial-analysis-benchmarks?slug=${encodeURIComponent(canonicalSlug)}`,
    REVAL_BENCH,
    {},
  );
  const rec = data.data?.[0];
  if (!rec) return null;
  const e = rec.benchmark_data?.evaluations ?? {};
  const p = rec.percentiles ?? {};
  const out: AABenchmark = {
    intelligence: e.artificial_analysis_intelligence_index ?? null,
    coding: e.artificial_analysis_coding_index ?? null,
    agentic: e.artificial_analysis_agentic_index ?? null,
    gpqa: e.gpqa ?? null,
    hle: e.hle ?? null,
    scicode: e.scicode ?? null,
    tau2: e.tau2 ?? null,
    intelligence_pct: p.intelligence_percentile ?? null,
    coding_pct: p.coding_percentile ?? null,
    agentic_pct: p.agentic_percentile ?? null,
  };
  return Object.values(out).some((v) => v !== null) ? out : null;
}

interface DARecord {
  category?: string;
  elo?: number;
  win_rate?: number;
  elo_percentile?: number;
  total_tournaments?: number;
}

export async function fetchDA(canonicalSlug: string): Promise<DABenchmark | null> {
  const data = await getJSON<{ data?: { records?: DARecord[] } }>(
    `${INTERNAL}/design-arena-benchmarks?slug=${encodeURIComponent(canonicalSlug)}`,
    REVAL_BENCH,
    {},
  );
  const recs = data.data?.records ?? [];
  if (!recs.length) return null;
  const best = recs.reduce((a, b) => ((b.elo ?? 0) > (a.elo ?? 0) ? b : a));
  const categories: DABenchmark["categories"] = {};
  for (const r of recs) {
    if (r.category)
      categories[r.category] = { elo: r.elo ?? null, win_rate: r.win_rate ?? null };
  }
  return {
    elo: best.elo ?? null,
    category: best.category ?? null,
    win_rate: best.win_rate ?? null,
    elo_pct: best.elo_percentile ?? null,
    tournaments: best.total_tournaments ?? null,
    categories,
  };
}

// ---- provider details (per-model page) --------------------------------------

interface RawEndpoint {
  provider_name?: string;
  quantization?: string;
  context_length?: number;
  max_completion_tokens?: number | null;
  uptime_last_30m?: number | null;
  supports_implicit_caching?: boolean;
  pricing?: Record<string, string>;
}

export async function fetchProviderDetails(
  modelId: string,
): Promise<ProviderDetail[]> {
  const data = await getJSON<{ data?: { endpoints?: RawEndpoint[] } }>(
    `${BASE}/models/${modelId}/endpoints`,
    REVAL_LIVE,
    {},
  );
  const endpoints = data.data?.endpoints ?? [];
  return endpoints.map((ep) => ({
    name: ep.provider_name ?? "Unknown",
    quantization: ep.quantization ?? "unknown",
    context_length: ep.context_length ?? 0,
    max_completion: ep.max_completion_tokens ?? null,
    price_input: perMillion(ep.pricing?.prompt),
    price_output: perMillion(ep.pricing?.completion),
    uptime: ep.uptime_last_30m != null ? Math.round(ep.uptime_last_30m * 10) / 10 : null,
    caching: Boolean(ep.supports_implicit_caching),
  }));
}

// ---- base catalog ------------------------------------------------------------

export async function fetchRawModels(): Promise<RawModel[]> {
  const data = await getJSON<{ data?: RawModel[] }>(
    `${BASE}/models`,
    REVAL_LIVE,
    {},
  );
  return (data.data ?? []).filter(
    (m) =>
      // Drop OpenRouter meta-routers and "~…-latest" alias entries that
      // duplicate a concrete dated model.
      !m.id.startsWith("openrouter/") && !m.id.startsWith("~"),
  );
}

// Total distinct providers across OpenRouter (one cheap call) — used for the
// homepage KPI without needing per-model endpoint fetches.
export async function fetchProviderCount(): Promise<number> {
  const data = await getJSON<{ data?: unknown[] }>(
    "https://openrouter.ai/api/frontend/all-providers",
    REVAL_LIVE,
    {},
  );
  return data.data?.length ?? 0;
}

export function baseModel(m: RawModel): Model {
  const arch = m.architecture ?? {};
  const top = m.top_provider ?? {};
  const p = m.pricing ?? {};
  return {
    id: m.id,
    canonical_slug: m.canonical_slug ?? m.id,
    name: m.name ?? m.id,
    description: (m.description ?? "").slice(0, 1400),
    context_length: m.context_length ?? 0,
    max_output: top.max_completion_tokens ?? null,
    throughput: 0,
    latency: null,
    uptime: null,
    price_input: perMillion(p.prompt),
    price_output: perMillion(p.completion),
    price_cache_read: perMillion(p.input_cache_read),
    supports_caching: false,
    providers: [],
    modality: arch.modality ?? "text->text",
    input_modalities: arch.input_modalities ?? ["text"],
    output_modalities: arch.output_modalities ?? ["text"],
    capabilities: capabilities(m),
    knowledge_cutoff: m.knowledge_cutoff ?? null,
    is_moderated: top.is_moderated ?? null,
    hugging_face_id: m.hugging_face_id ?? null,
    created: m.created ?? 0,
    aa: null,
    da: null,
  };
}

// Bounded-concurrency map to avoid hammering OpenRouter with 1000+ parallel reqs.
export async function pool<T, R>(
  items: T[],
  limit: number,
  fn: (item: T, i: number) => Promise<R>,
): Promise<R[]> {
  const results = new Array<R>(items.length);
  let next = 0;
  async function worker() {
    while (next < items.length) {
      const i = next++;
      results[i] = await fn(items[i], i);
    }
  }
  await Promise.all(
    Array.from({ length: Math.min(limit, items.length) }, worker),
  );
  return results;
}
