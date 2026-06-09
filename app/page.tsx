import { getCatalog } from "@/lib/catalog";
import { fetchProviderCount } from "@/lib/openrouter";
import { Dashboard } from "@/components/Dashboard";
import { Footer } from "@/components/Footer";
import { TopChart } from "@/components/TopChart";

export const revalidate = 3600;

function StatPill({ value, label }: { value: string | number; label: string }) {
  return (
    <div className="flex items-baseline gap-1.5">
      <span className="text-lg font-bold tracking-tight text-ink">{value}</span>
      <span className="text-[13px] text-ink-3">{label}</span>
    </div>
  );
}

export default async function Home() {
  const [models, providerCount] = await Promise.all([getCatalog(), fetchProviderCount()]);
  const benchmarked = models.filter((m) => m.aa || m.da).length;
  const stats = { models: models.length, providers: providerCount, benchmarked };

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
        <div className="mx-auto w-full max-w-[1320px] px-5 pb-9 pt-7">
          <span className="font-mono text-[15px] font-bold tracking-tight text-ink">
            model<span className="text-brand">grep</span>
          </span>
        </div>
        <div className="mx-auto grid w-full max-w-[1320px] items-center gap-8 px-5 pb-10 lg:grid-cols-[1fr_minmax(0,480px)]">
          <div>
            <div className="mb-3 inline-flex items-center gap-1.5 rounded-full border border-line bg-surface px-2.5 py-1 text-[11px] font-medium text-ink-2">
              <span className="size-1.5 animate-pulse rounded-full bg-elite" />
              Live benchmarks · updated hourly
            </div>
            <h1 className="text-[32px] font-bold leading-[1.1] tracking-tight text-ink sm:text-[40px]">
              Find &amp; understand
              <br />
              every LLM.
            </h1>
            <p className="mt-3 max-w-md text-[15px] leading-relaxed text-ink-2">
              The leaderboard for AI models — ranked by real intelligence benchmarks, speed, and price.
              Compare, filter, and dig into any model.
            </p>
            <div className="mt-6 flex flex-wrap items-center gap-x-6 gap-y-2">
              <StatPill value={stats.models} label="models" />
              <StatPill value={stats.providers} label="providers" />
              <StatPill value={stats.benchmarked} label="benchmarked" />
            </div>
          </div>

          <TopChart models={models} />
        </div>
      </section>

      <div className="pt-5">
        <Dashboard models={models} providers={[]} stats={stats} />
      </div>

      <Footer />
    </div>
  );
}
