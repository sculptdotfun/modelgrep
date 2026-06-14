import Link from "next/link";
import { getCatalog, toLite } from "@/lib/catalog";
import { fetchProviderCount } from "@/lib/openrouter";
import { Dashboard } from "@/components/Dashboard";
import { Footer } from "@/components/Footer";
import { SiteHeader } from "@/components/SiteHeader";
import { TopChart } from "@/components/TopChart";
import { OwnerAvatar } from "@/components/ui";
import { COLLECTIONS } from "@/lib/collections";
import { resolveFacet } from "@/lib/facets";
import { modelOwner } from "@/lib/format";
import type { Model } from "@/lib/types";

export const revalidate = 3600;

// One-line intent descriptor per ranking — the "what am I optimizing for" hook
// that turns a bare collection title into a scannable decision.
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
};

// "Browse by need" — an intent-based entry layer between the hero and the dense
// 300-row table. Each card surfaces the current #1 model for a ranking, so a
// visitor lands on an answer instead of a wall of rows. Doubles as a keyword-rich
// internal-link mesh into the /best collections.
function BrowseByNeed({ models }: { models: Model[] }) {
  // Lead with the "small & fast" tier — it's the most-searched, least-served
  // intent ("small fast anthropic model") and lives outside COLLECTIONS.
  const smallFacet = resolveFacet(["small"], models);
  const smallCard =
    smallFacet?.models[0]
      ? [{ c: { slug: "small", title: "Small & Fast LLMs", metricLabel: "Speed", display: smallFacet.base.display }, top: smallFacet.models[0] }]
      : [];
  const cards = [
    ...smallCard,
    ...COLLECTIONS.map((c) => {
      const top = models.filter(c.filter).sort(c.sort)[0];
      return top ? { c: { slug: c.slug, title: c.title, metricLabel: c.metricLabel, display: c.display }, top } : null;
    }).filter((x): x is { c: { slug: string; title: string; metricLabel: string; display: (m: Model) => string }; top: Model } => x !== null),
  ];

  return (
    <section className="mx-auto w-full max-w-[1200px] px-5 pb-2 pt-10">
      <div className="mb-4 flex items-baseline justify-between">
        <h2 className="font-display text-[20px] font-bold tracking-tight text-ink">Browse by need</h2>
        <span className="text-[12px] text-ink-3">Pick what you&apos;re optimizing for</span>
      </div>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {cards.map(({ c, top }) => {
          const shortName = top.name.includes(": ") ? top.name.split(": ").slice(1).join(": ") : top.name;
          return (
            <Link
              key={c.slug}
              href={`/best/${c.slug}`}
              className="card-shadow card-lift group flex items-center gap-3.5 rounded-lg border border-line bg-surface p-4"
            >
              <OwnerAvatar owner={modelOwner(top.id)} size={36} />
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="text-[14px] font-semibold text-ink group-hover:text-brand-ink">{c.title}</span>
                </div>
                <div className="mt-0.5 truncate text-[12px] text-ink-3">{NEED_HINT[c.slug] ?? c.metricLabel}</div>
              </div>
              <div className="shrink-0 text-right">
                <div className="truncate font-mono text-[12px] font-medium text-ink-2 group-hover:text-ink">#1 {shortName}</div>
                <div className="font-mono text-[13px] font-bold tabular-nums text-ink">{c.display(top)}</div>
              </div>
            </Link>
          );
        })}
      </div>
    </section>
  );
}

function StatPill({ value, label, first }: { value: string | number; label: string; first?: boolean }) {
  return (
    <div className={`flex flex-col py-3 pr-8 ${first ? "" : "pl-8"}`}>
      <span className="font-mono text-[22px] font-bold leading-none tracking-tight text-ink">{value}</span>
      <span className="mt-1.5 font-mono text-[10px] uppercase tracking-widest text-ink-3">{label}</span>
    </div>
  );
}

export default async function Home() {
  const [models, providerCount] = await Promise.all([getCatalog(), fetchProviderCount()]);
  const benchmarked = models.filter((m) => m.aa || m.da).length;
  const stats = { models: models.length, providers: providerCount, benchmarked };
  const lite = models.map(toLite);

  const top = models
    .filter((m) => m.aa?.intelligence != null)
    .sort((a, b) => b.aa!.intelligence! - a.aa!.intelligence!)
    .slice(0, 15);
  const itemListJsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: "LLM leaderboard ranked by intelligence",
    description: "AI models ranked by the Artificial Analysis Intelligence Index.",
    numberOfItems: top.length,
    itemListElement: top.map((m, i) => ({
      "@type": "ListItem",
      position: i + 1,
      name: m.name,
      url: `https://modelgrep.com/models/${m.id}`,
    })),
  };

  return (
    <div className="min-h-screen">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(itemListJsonLd) }} />
      <section className="border-b border-line bg-surface">
        <div className="mx-auto w-full max-w-[1200px] px-5 pb-8 pt-6">
          <SiteHeader />
        </div>
        <div className="mx-auto grid w-full max-w-[1200px] items-center gap-10 px-5 pb-12 lg:grid-cols-[1fr_minmax(0,480px)]">
          <div>
            <div className="mb-4 flex items-center gap-2 font-mono text-[11px] font-medium uppercase tracking-widest text-ink-3">
              <span className="size-1.5 rounded-full bg-elite" />
              Live · updated hourly
            </div>
            <h1 className="font-display text-[38px] font-bold leading-[1.02] text-ink sm:text-[48px]">
              Find &amp; understand
              <br />
              every LLM.
            </h1>
            <p className="mt-4 max-w-md text-[15px] leading-relaxed text-ink-2">
              The LLM leaderboard — compare {stats.models}+ AI models by intelligence benchmark, speed, latency,
              price and context. Find the smartest, fastest, cheapest, or smallest model for the job.
            </p>
            <div className="mt-7 flex items-stretch divide-x divide-line border-y border-line">
              <StatPill value={stats.models} label="models" first />
              <StatPill value={stats.providers} label="providers" />
              <StatPill value={stats.benchmarked} label="benchmarked" />
            </div>
            <div className="mt-5 flex flex-wrap gap-1.5">
              <Link
                href="/best"
                className="rounded-md border border-ink bg-ink px-2.5 py-1 text-[12px] font-medium text-white transition-opacity hover:opacity-90"
              >
                All rankings
              </Link>
              <Link
                href="/best/small"
                className="rounded-md border border-line px-2.5 py-1 text-[12px] font-medium text-ink-2 transition-colors hover:border-line-strong hover:text-ink"
              >
                Small &amp; Fast
              </Link>
              {COLLECTIONS.slice(0, 4).map((c) => (
                <Link
                  key={c.slug}
                  href={`/best/${c.slug}`}
                  className="rounded-md border border-line px-2.5 py-1 text-[12px] font-medium text-ink-2 transition-colors hover:border-line-strong hover:text-ink"
                >
                  {c.title}
                </Link>
              ))}
            </div>
          </div>

          <TopChart models={lite} />
        </div>
      </section>

      <BrowseByNeed models={models} />

      <div className="mx-auto w-full max-w-[1200px] px-5 pb-3 pt-10">
        <div className="flex items-baseline justify-between border-b border-line pb-3">
          <h2 className="font-display text-[20px] font-bold tracking-tight text-ink">All models</h2>
          <span className="text-[12px] text-ink-3">{stats.models} tracked · search, filter &amp; sort</span>
        </div>
      </div>

      <div className="pt-1">
        <Dashboard models={lite} providers={[]} stats={stats} />
      </div>

      <Footer />
    </div>
  );
}
