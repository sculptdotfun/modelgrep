import Link from "next/link";
import {
  ArrowRight,
  Bot,
  Brain,
  Code2,
  DollarSign,
  Eye,
  Gauge,
  Gift,
  Layers,
  type LucideIcon,
  Palette,
  ScrollText,
  Sparkles,
  Timer,
  Zap,
} from "lucide-react";
import { getCatalog, toLite } from "@/lib/catalog";
import { fetchProviderCount } from "@/lib/openrouter";
import { Dashboard } from "@/components/Dashboard";
import { Footer } from "@/components/Footer";
import { TopChart } from "@/components/TopChart";
import { COLLECTIONS } from "@/lib/collections";
import { resolveFacet } from "@/lib/facets";
import { modelOwner } from "@/lib/format";
import { ownerColor } from "@/lib/owners";
import type { Model } from "@/lib/types";

export const revalidate = 3600;

// One-line intent descriptor + icon per ranking — turns a bare collection title
// into a scannable, iconic decision.
const NEED: Record<string, { hint: string; Icon: LucideIcon }> = {
  small: { hint: "Cheap, fast, efficient tier", Icon: Zap },
  smartest: { hint: "Highest intelligence index", Icon: Brain },
  coding: { hint: "Best at real software tasks", Icon: Code2 },
  cheapest: { hint: "Lowest price per token", Icon: DollarSign },
  fastest: { hint: "Most tokens per second", Icon: Gauge },
  reasoning: { hint: "Deepest step-by-step thinking", Icon: Sparkles },
  agents: { hint: "Best at tool use & planning", Icon: Bot },
  "long-context": { hint: "Largest context window", Icon: ScrollText },
  vision: { hint: "Reads images & documents", Icon: Eye },
  free: { hint: "Capable at zero cost", Icon: Gift },
  "lowest-latency": { hint: "Fastest to first token", Icon: Timer },
  "open-source": { hint: "Self-hostable open weights", Icon: Layers },
  design: { hint: "Best UI & frontend output", Icon: Palette },
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
    <section className="mx-auto w-full max-w-[1200px] px-5 pb-2 pt-12">
      <div className="mb-5 flex items-end justify-between">
        <div>
          <h2 className="font-display text-[22px] font-bold tracking-tight text-ink">Browse by need</h2>
          <p className="mt-1 text-[13px] text-ink-3">Pick what you&apos;re optimizing for — each lands on a ranked answer.</p>
        </div>
      </div>
      <div className="grid grid-cols-1 gap-3.5 sm:grid-cols-2 lg:grid-cols-3">
        {cards.map(({ c, top }) => {
          const shortName = top.name.includes(": ") ? top.name.split(": ").slice(1).join(": ") : top.name;
          const meta = NEED[c.slug];
          const Icon = meta?.Icon ?? Sparkles;
          return (
            <Link
              key={c.slug}
              href={`/best/${c.slug}`}
              className="card-shadow card-lift group rounded-2xl border border-line bg-surface p-4"
            >
              <div className="flex items-center gap-3">
                <span className="flex size-9 items-center justify-center rounded-xl bg-surface-2 text-ink-2 transition-colors group-hover:bg-brand/10 group-hover:text-brand">
                  <Icon className="size-[18px]" strokeWidth={2} />
                </span>
                <div className="min-w-0 flex-1">
                  <div className="text-[14px] font-semibold text-ink group-hover:text-brand-ink">{c.title}</div>
                  <div className="mt-0.5 truncate text-[12px] text-ink-3">{meta?.hint ?? c.metricLabel}</div>
                </div>
                <ArrowRight className="size-4 -translate-x-1 text-ink-3 opacity-0 transition-all group-hover:translate-x-0 group-hover:opacity-100" strokeWidth={2} />
              </div>
              <div className="mt-3.5 flex items-center justify-between border-t border-line pt-3">
                <span className="flex min-w-0 items-center gap-1.5">
                  <span className="size-2 shrink-0 rounded-[3px]" style={{ background: ownerColor(modelOwner(top.id)) }} />
                  <span className="truncate font-mono text-[11.5px] text-ink-2">#1 {shortName}</span>
                </span>
                <span className="shrink-0 font-mono text-[13px] font-bold tabular-nums text-ink">{c.display(top)}</span>
              </div>
            </Link>
          );
        })}
      </div>
    </section>
  );
}

function StatPill({ value, label }: { value: string | number; label: string }) {
  return (
    <div className="card-shadow flex flex-col rounded-xl border border-line bg-surface px-4 py-3">
      <span className="font-mono text-[23px] font-bold leading-none tracking-tight text-ink">{value}</span>
      <span className="mt-2 text-[10px] uppercase tracking-widest text-ink-3">{label}</span>
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
      <section className="hero-glow relative border-b border-line">
        <div className="pointer-events-none absolute inset-0 dot-grid opacity-[0.35] [mask-image:radial-gradient(80%_60%_at_50%_0%,#000,transparent)]" />
        <div className="relative mx-auto grid w-full max-w-[1200px] items-center gap-10 px-5 pb-16 pt-14 lg:grid-cols-[1fr_minmax(0,480px)]">
          <div>
            <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-line bg-surface/70 py-1 pl-2 pr-3 text-[11px] font-medium text-ink-2 backdrop-blur">
              <span className="relative flex size-2">
                <span className="absolute inline-flex size-full animate-ping rounded-full bg-elite opacity-60" />
                <span className="relative inline-flex size-2 rounded-full bg-elite" />
              </span>
              Live · updated hourly
            </div>
            <h1 className="font-display text-[40px] font-bold leading-[1.0] tracking-tight text-ink sm:text-[52px]">
              Find &amp; understand
              <br />
              <span className="gradient-ink">every LLM.</span>
            </h1>
            <p className="mt-5 max-w-md text-[15.5px] leading-relaxed text-ink-2">
              The LLM leaderboard — compare {stats.models}+ AI models by intelligence benchmark, speed, latency,
              price and context. Find the smartest, fastest, cheapest, or smallest model for the job.
            </p>
            <div className="mt-7 grid max-w-md grid-cols-3 gap-3">
              <StatPill value={stats.models} label="models" />
              <StatPill value={stats.providers} label="providers" />
              <StatPill value={stats.benchmarked} label="benchmarked" />
            </div>
            <div className="mt-6 flex flex-wrap items-center gap-2">
              <Link href="/best" className="btn btn-primary h-9 px-4">
                <Layers className="size-4" strokeWidth={2} />
                All rankings
              </Link>
              <Link href="/best/small" className="btn btn-secondary h-9 px-3.5">
                <Zap className="size-[15px] text-brand" strokeWidth={2} />
                Small &amp; Fast
              </Link>
              {COLLECTIONS.slice(0, 3).map((c) => (
                <Link key={c.slug} href={`/best/${c.slug}`} className="btn btn-secondary h-9 px-3.5">
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
