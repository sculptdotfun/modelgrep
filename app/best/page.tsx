import type { Metadata } from "next";
import Link from "next/link";
import { getCatalog } from "@/lib/catalog";
import { COLLECTIONS } from "@/lib/collections";
import { resolveFacet, makerOptions } from "@/lib/facets";
import { AnswerBox } from "@/components/AnswerBox";
import { Footer } from "@/components/Footer";
import { OwnerAvatar } from "@/components/ui";
import { fmtMonth, fmtPrice, fmtThroughput, modelOwner } from "@/lib/format";
import type { Model } from "@/lib/types";

export const revalidate = 3600;

export const metadata: Metadata = {
  title: "Best LLMs — Rankings by Intelligence, Speed, Price & Use Case",
  description:
    "Every LLM ranking in one place: the smartest, fastest, cheapest, best for coding, agents, reasoning and vision — plus the best model from each maker (Anthropic, OpenAI, Google and more). Data-backed and updated continuously.",
  keywords: ["best LLM", "best AI model", "LLM rankings", "best LLM for coding", "fastest LLM", "cheapest LLM"],
  alternates: { canonical: "/best" },
  openGraph: { title: "Best LLMs — Rankings by Intelligence, Speed, Price & Use Case", description: "The smartest, fastest, cheapest and best-for-X AI models, ranked.", url: "/best", type: "website" },
};

// High-intent maker × ranking combos to feature on the hub.
const FEATURED: [string, string][] = [
  ["small", "anthropic"],
  ["cheapest", "openai"],
  ["fastest", "anthropic"],
  ["coding", "anthropic"],
  ["smartest", "google"],
  ["cheapest", "google"],
  ["fastest", "openai"],
  ["reasoning", "deepseek"],
  ["small", "google"],
  ["coding", "openai"],
  ["cheapest", "anthropic"],
  ["smartest", "openai"],
];

const NEED_HINT: Record<string, string> = {
  smartest: "Highest intelligence index",
  coding: "Best at real software tasks",
  cheapest: "Lowest price per token",
  fastest: "Most tokens per second",
  reasoning: "Deepest step-by-step thinking",
  agents: "Best at tool use & planning",
  "long-context": "Largest context window",
  vision: "Reads images & documents",
  free: "Capable at zero cost",
  "lowest-latency": "Fastest to first token",
  "open-source": "Self-hostable open weights",
  design: "Best UI & frontend output",
  small: "Cheap, fast, efficient tier",
};

function topFor(slug: string, catalog: Model[]): Model | null {
  return resolveFacet([slug], catalog)?.models[0] ?? null;
}

export default async function BestHub() {
  const catalog = await getCatalog();
  const updated = fmtMonth(new Date());

  const smartest = topFor("smartest", catalog);
  const fastest = topFor("fastest", catalog);
  const cheapest = topFor("cheapest", catalog);
  const answer = [
    smartest && `The smartest LLM right now is ${smartest.name.replace(/^.*: /, "")} (${smartest.aa?.intelligence?.toFixed(1)} intelligence)`,
    fastest && `the fastest is ${fastest.name.replace(/^.*: /, "")} at ${fmtThroughput(fastest.throughput)} tokens/sec`,
    cheapest && `the cheapest is ${cheapest.name.replace(/^.*: /, "")} at ${fmtPrice(cheapest.price_input)} per million input tokens`,
  ]
    .filter(Boolean)
    .join(", ")
    .replace(/,([^,]*)$/, ", and$1") + ". Browse every ranking below, or narrow any of them to a single maker.";

  const rankings = [{ slug: "small", title: "Small & Fast LLMs" }, ...COLLECTIONS.map((c) => ({ slug: c.slug, title: c.title }))];
  const featured = FEATURED.map(([base, mk]) => resolveFacet([base, mk], catalog)).filter((x): x is NonNullable<typeof x> => x != null);
  const makers = makerOptions(catalog).slice(0, 14);

  return (
    <div className="min-h-screen">
      <div className="mx-auto w-full max-w-[1200px] px-5 py-7">

        <h1 className="font-display mt-7 text-[32px] font-bold leading-tight text-ink sm:text-[38px]">Best LLMs, ranked</h1>
        <AnswerBox answer={answer} updated={updated} />

        {/* Rankings by metric */}
        <section className="mt-9">
          <h2 className="mb-3 font-display text-[20px] font-bold tracking-tight text-ink">By metric &amp; use case</h2>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {rankings.map((r) => {
              const top = topFor(r.slug, catalog);
              return (
                <Link key={r.slug} href={`/best/${r.slug}`} className="card-shadow card-lift group flex items-center gap-3.5 rounded-lg border border-line bg-surface p-4">
                  {top && <OwnerAvatar owner={modelOwner(top.id)} size={34} />}
                  <div className="min-w-0 flex-1">
                    <div className="text-[14px] font-semibold text-ink group-hover:text-brand-ink">{r.title}</div>
                    <div className="mt-0.5 truncate text-[12px] text-ink-3">{NEED_HINT[r.slug]}</div>
                  </div>
                  {top && <span className="shrink-0 truncate font-mono text-[11px] text-ink-3">#1 {top.name.replace(/^.*: /, "")}</span>}
                </Link>
              );
            })}
          </div>
        </section>

        {/* Featured intersections */}
        {featured.length > 0 && (
          <section className="mt-10">
            <h2 className="mb-3 font-display text-[20px] font-bold tracking-tight text-ink">Popular maker rankings</h2>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {featured.map((f) => {
                const top = f.models[0];
                return (
                  <Link key={f.canonical} href={f.canonical} className="card-shadow card-lift group flex items-center gap-3.5 rounded-lg border border-line bg-surface p-4">
                    <OwnerAvatar owner={modelOwner(top.id)} size={34} />
                    <div className="min-w-0 flex-1">
                      <div className="text-[14px] font-semibold text-ink group-hover:text-brand-ink">{f.h1}</div>
                      <div className="mt-0.5 truncate font-mono text-[11px] text-ink-3">#1 {top.name.replace(/^.*: /, "")} · {f.base.display(top)}</div>
                    </div>
                  </Link>
                );
              })}
            </div>
          </section>
        )}

        {/* By maker */}
        <section className="mt-10">
          <h2 className="mb-3 font-display text-[20px] font-bold tracking-tight text-ink">By maker</h2>
          <div className="flex flex-wrap gap-2">
            {makers.map((mk) => (
              <Link key={mk.slug} href={`/makers/${mk.slug}`} className="flex items-center gap-2 rounded-md border border-line bg-surface px-3 py-1.5 text-[13px] text-ink-2 hover:text-brand-ink">
                <OwnerAvatar owner={mk.slug} size={18} />
                {mk.displayName}
              </Link>
            ))}
          </div>
        </section>
      </div>
      <Footer />
    </div>
  );
}
