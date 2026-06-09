import Link from "next/link";
import { getCatalog, toLite } from "@/lib/catalog";
import { fetchProviderCount } from "@/lib/openrouter";
import { Dashboard } from "@/components/Dashboard";
import { Footer } from "@/components/Footer";
import { SiteHeader } from "@/components/SiteHeader";
import { TopChart } from "@/components/TopChart";
import { COLLECTIONS } from "@/lib/collections";

export const revalidate = 3600;

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
        <div className="mx-auto w-full max-w-[1320px] px-5 pb-8 pt-6">
          <SiteHeader />
        </div>
        <div className="mx-auto grid w-full max-w-[1320px] items-center gap-10 px-5 pb-12 lg:grid-cols-[1fr_minmax(0,480px)]">
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
              The leaderboard for AI models — ranked by real intelligence benchmarks, speed, and price.
              Compare, filter, and dig into any model.
            </p>
            <div className="mt-7 flex items-stretch divide-x divide-line border-y border-line">
              <StatPill value={stats.models} label="models" first />
              <StatPill value={stats.providers} label="providers" />
              <StatPill value={stats.benchmarked} label="benchmarked" />
            </div>
            <div className="mt-5 flex flex-wrap gap-1.5">
              {COLLECTIONS.slice(0, 6).map((c) => (
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

      <div className="pt-5">
        <Dashboard models={lite} providers={[]} stats={stats} />
      </div>

      <Footer />
    </div>
  );
}
