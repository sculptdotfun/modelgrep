// Catalog aggregator — composes base models + live perf + benchmarks into the
// enriched array used by the dashboard and model pages. Cached so the heavy
// fan-out (perf RSC per model) runs once per revalidate window, not per request.

import { unstable_cache } from "next/cache";
import {
  baseModel,
  fetchAA,
  fetchDA,
  fetchPerf,
  fetchProviderDetails,
  fetchRawModels,
  pool,
} from "./openrouter";
import type { LiteModel, Model, ModelDetail } from "./types";

// Strip a catalog Model down to what client components actually render.
export function toLite(m: Model): LiteModel {
  return {
    id: m.id,
    name: m.name,
    context_length: m.context_length,
    throughput: m.throughput,
    latency: m.latency,
    price_input: m.price_input,
    price_output: m.price_output,
    providers: m.providers,
    capabilities: m.capabilities,
    aa: m.aa
      ? { intelligence: m.aa.intelligence, intelligence_pct: m.aa.intelligence_pct, coding: m.aa.coding }
      : null,
    da: m.da ? { elo: m.da.elo, category: m.da.category } : null,
  };
}

const CONCURRENCY = 24;

async function buildCatalog(): Promise<Model[]> {
  const raw = await fetchRawModels();
  const models = raw.map(baseModel);

  // Live perf (throughput/latency) per model — heavy RSC fetch, bounded pool.
  const perf = await pool(models, CONCURRENCY, (m) => fetchPerf(m.id));
  models.forEach((m, i) => {
    const p = perf[i];
    if (p) {
      m.throughput = p.throughput;
      m.latency = p.latency;
    }
  });

  // Benchmarks per unique canonical slug (cheap JSON, daily cache upstream).
  const slugs = Array.from(new Set(models.map((m) => m.canonical_slug)));
  const benches = await pool(slugs, CONCURRENCY, async (slug) => {
    const [aa, da] = await Promise.all([fetchAA(slug), fetchDA(slug)]);
    return [slug, { aa, da }] as const;
  });
  const benchBySlug = new Map(benches);
  for (const m of models) {
    const b = benchBySlug.get(m.canonical_slug);
    if (b) {
      m.aa = b.aa;
      m.da = b.da;
    }
  }

  return models;
}

export const getCatalog = unstable_cache(buildCatalog, ["modelgrep-catalog-v3"], {
  revalidate: 3600,
  tags: ["catalog"],
});

export async function getModel(slug: string): Promise<Model | undefined> {
  const catalog = await getCatalog();
  return catalog.find((m) => m.id === slug);
}

// Per-model page: enriched record + live provider breakdown.
export async function getModelDetail(
  slug: string,
): Promise<ModelDetail | undefined> {
  const base = await getModel(slug);
  if (!base) return undefined;
  const provider_details = await fetchProviderDetails(slug);
  const uptime = provider_details.reduce<number | null>(
    (best, p) => (p.uptime != null && (best == null || p.uptime > best) ? p.uptime : best),
    null,
  );
  return {
    ...base,
    provider_details,
    providers: provider_details.map((p) => p.name),
    uptime: uptime ?? base.uptime,
    supports_caching: provider_details.some((p) => p.caching) || base.supports_caching,
  };
}
