import Link from "next/link";
import { getCatalog, toLite } from "@/lib/catalog";
import { fetchProviderCount } from "@/lib/openrouter";
import { Dashboard } from "@/components/Dashboard";
import { Footer } from "@/components/Footer";
import { SiteHeader } from "@/components/SiteHeader";
import { TopChart } from "@/components/TopChart";
import { COLLECTIONS } from "@/lib/collections";

export const revalidate = 3600;

function StatPill({ value, label }: { value: string | number; label: string }) {
  return (
    <div className="flex items-baseline gap-1.5">
      <span className="font-display text-xl font-bold text-ink">{value}</span>
      <span className="text-[13px] text-ink-3">{label}</span>
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
      <section className="dot-texture border-b border-line">
        <div className="mx-auto w-full max-w-[1320px] px-5 pb-9 pt-6">
          <SiteHeader />
        </div>
        <div className="mx-auto grid w-full max-w-[1320px] items-center gap-8 px-5 pb-12 lg:grid-cols-[1fr_minmax(0,480px)]">
          <div>
            <div className="mb-4 inline-flex items-center gap-1.5 rounded-full border border-line bg-surface px-2.5 py-1 text-[11px] font-medium text-ink-2">
              <span className="size-1.5 animate-pulse rounded-full bg-elite" />
              Live benchmarks · updated hourly
            </div>
            <h1 className="font-display text-[36px] font-bold leading-[1.04] text-ink sm:text-[46px]">
              Find &amp; understand
              <br />
              <span className="text-gradient">every LLM.</span>
            </h1>
            <p className="mt-4 max-w-md text-[15px] leading-relaxed text-ink-2">
              The leaderboard for AI models — ranked by real intelligence benchmarks, speed, and price.
              Compare, filter, and dig into any model.
            </p>
            <div className="mt-6 flex flex-wrap items-center gap-x-6 gap-y-2">
              <StatPill value={stats.models} label="models" />
              <StatPill value={stats.providers} label="providers" />
              <StatPill value={stats.benchmarked} label="benchmarked" />
            </div>
            <div className="mt-5 flex flex-wrap gap-1.5">
              {COLLECTIONS.slice(0, 6).map((c) => (
                <Link
                  key={c.slug}
                  href={`/best/${c.slug}`}
                  className="rounded-full border border-line bg-surface px-2.5 py-1 text-[12px] text-ink-2 transition-colors hover:text-brand-ink"
                >
                  {c.title}
                </Link>
              ))}
            </div>
          </div>

          <TopChart models={lite} />
        </div>
      </section>

      <div className="pt-5">
        <Dashboard models={lite} providers={[]} stats={stats} />
      </div>

      <Footer />
    </div>
  );
}
