import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { getCatalog } from "@/lib/catalog";
import { groupByMaker } from "@/lib/makers";
import { AnswerBox } from "@/components/AnswerBox";
import { Footer } from "@/components/Footer";
import { CapBadges, OwnerAvatar } from "@/components/ui";
import { fmtContext, fmtLatency, fmtMonth, fmtPrice, fmtThroughput } from "@/lib/format";

// Maker-scoped ranking shortcuts — the intersection pages that capture
// "fastest <maker> model", "cheapest <maker> model", etc.
const MAKER_FACETS: { slug: string; label: (n: string) => string }[] = [
  { slug: "smartest", label: (n) => `Smartest ${n}` },
  { slug: "fastest", label: (n) => `Fastest ${n}` },
  { slug: "cheapest", label: (n) => `Cheapest ${n}` },
  { slug: "small", label: (n) => `Small & fast ${n}` },
  { slug: "coding", label: (n) => `${n} for coding` },
  { slug: "reasoning", label: (n) => `${n} reasoning` },
];

function shortName(name: string): string {
  return name.includes(": ") ? name.split(": ").slice(1).join(": ") : name;
}

export const revalidate = 3600;

type Params = { slug: string };

export async function generateStaticParams() {
  const makers = groupByMaker(await getCatalog());
  return makers.slice(0, 30).map((mk) => ({ slug: mk.slug }));
}

async function getMaker(slug: string) {
  const makers = groupByMaker(await getCatalog());
  return makers.find((mk) => mk.slug === slug);
}

export async function generateMetadata({ params }: { params: Promise<Params> }): Promise<Metadata> {
  const { slug } = await params;
  const mk = await getMaker(slug);
  if (!mk) return { title: "Maker not found" };
  const title = `${mk.displayName} Models — All ${mk.models.length} Ranked & Benchmarked`;
  const description = `Every ${mk.displayName} AI model compared: intelligence benchmarks, speed, latency, context window and pricing for all ${mk.models.length} ${mk.displayName} models${mk.bestIntel ? `, led by ${mk.bestIntel.name}` : ""}. Updated continuously.`;
  return {
    title,
    description,
    keywords: [`${mk.displayName} models`, `${mk.displayName} AI models`, `${mk.displayName} LLM`, `${mk.displayName} model list`, `${mk.displayName} pricing`],
    alternates: { canonical: `/makers/${mk.slug}` },
    openGraph: { title, description, url: `/makers/${mk.slug}`, type: "article" },
    twitter: { card: "summary_large_image", title, description },
  };
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-lg border border-line bg-surface p-3.5">
      <div className="text-[10px] font-semibold uppercase tracking-wider text-ink-3">{label}</div>
      <div className="font-mono mt-1.5 text-[20px] font-bold leading-none tracking-tight text-ink">{value}</div>
      {sub && <div className="mt-1.5 truncate font-mono text-[11px] text-ink-3">{sub}</div>}
    </div>
  );
}

export default async function MakerPage({ params }: { params: Promise<Params> }) {
  const { slug } = await params;
  const mk = await getMaker(slug);
  if (!mk) notFound();

  const updated = fmtMonth(new Date());

  // BLUF answer — names the maker's standout model on each axis with real numbers.
  const answerParts: string[] = [];
  if (mk.bestIntel?.aa?.intelligence != null)
    answerParts.push(`${mk.displayName}'s smartest model is ${shortName(mk.bestIntel.name)} (${mk.bestIntel.aa.intelligence.toFixed(1)} on the Intelligence Index)`);
  if (mk.fastest)
    answerParts.push(`its fastest is ${shortName(mk.fastest.name)} at ${fmtThroughput(mk.fastest.throughput)} tokens/sec`);
  if (mk.cheapest)
    answerParts.push(`its cheapest is ${shortName(mk.cheapest.name)} at ${fmtPrice(mk.cheapest.price_input)} per million input tokens`);
  const answer =
    (answerParts.join(", ").replace(/,([^,]*)$/, ", and$1") || `${mk.displayName} has ${mk.models.length} models tracked on modelgrep`) +
    `. All ${mk.models.length} ${mk.displayName} models are compared below by intelligence, speed, latency, context and price.`;

  const jsonLd = [
    {
      "@context": "https://schema.org",
      "@type": "CollectionPage",
      name: `${mk.displayName} models`,
      url: `https://modelgrep.com/makers/${mk.slug}`,
      description: answer,
      dateModified: new Date().toISOString(),
    },
    {
      "@context": "https://schema.org",
      "@type": "ItemList",
      name: `${mk.displayName} models`,
      numberOfItems: mk.models.length,
      itemListElement: mk.models.slice(0, 25).map((m, i) => ({
        "@type": "ListItem",
        position: i + 1,
        name: m.name,
        url: `https://modelgrep.com/models/${m.id}`,
      })),
    },
  ];

  return (
    <div className="min-h-screen">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }} />
      <div className="mx-auto w-full max-w-[1200px] px-5 py-7">

        <nav className="mt-7 text-xs text-ink-3">
          <Link href="/" className="hover:text-ink-2">
            models
          </Link>
          <span className="mx-1.5">/</span>
          <span className="text-ink">{mk.slug}</span>
        </nav>

        <div className="mt-4 flex items-center gap-4">
          <OwnerAvatar owner={mk.slug} size={52} />
          <div>
            <h1 className="font-display text-[30px] font-bold leading-tight text-ink">{mk.displayName} models</h1>
            <p className="mt-1 text-[14px] text-ink-2">
              {mk.models.length} models tracked · ranked by intelligence, speed &amp; price
            </p>
          </div>
        </div>

        <AnswerBox answer={answer} updated={updated} />

        <div className="mt-5 flex flex-wrap gap-2">
          {MAKER_FACETS.map((fct) => (
            <Link
              key={fct.slug}
              href={`/best/${fct.slug}/${mk.slug}`}
              className="rounded-md border border-line bg-surface px-3 py-1.5 text-[12px] font-medium text-ink-2 transition-colors hover:border-line-strong hover:text-brand-ink"
            >
              {fct.label(mk.displayName)}
            </Link>
          ))}
        </div>

        <div className="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-3">
          <Stat
            label="Smartest"
            value={mk.bestIntel?.aa?.intelligence != null ? mk.bestIntel.aa.intelligence.toFixed(1) : "—"}
            sub={mk.bestIntel?.id}
          />
          <Stat label="Fastest" value={mk.fastest ? `${fmtThroughput(mk.fastest.throughput)} t/s` : "—"} sub={mk.fastest?.id} />
          <Stat label="Cheapest" value={mk.cheapest ? `${fmtPrice(mk.cheapest.price_input)}/M` : "—"} sub={mk.cheapest?.id} />
        </div>

        <div className="mt-7 overflow-hidden rounded-lg border border-line bg-surface">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-line text-[10px] uppercase tracking-wider text-ink-3">
                <th className="px-4 py-2.5 text-left font-semibold">Model</th>
                <th className="px-3 py-3.5 text-right font-semibold">Intel</th>
                <th className="hidden px-3 py-3.5 text-right font-semibold sm:table-cell">Speed</th>
                <th className="hidden px-3 py-3.5 text-right font-semibold sm:table-cell">Latency</th>
                <th className="px-3 py-3.5 text-right font-semibold">In $/M</th>
                <th className="px-4 py-3.5 text-right font-semibold">Context</th>
              </tr>
            </thead>
            <tbody>
              {mk.models.map((m) => (
                <tr key={m.id} className="border-b border-line transition-colors last:border-0 hover:bg-surface-2/60">
                  <td className="px-4 py-3.5">
                    <Link href={`/models/${m.id}`} className="font-mono text-[13px] font-medium text-ink hover:text-brand-ink">
                      {m.id.split("/").slice(1).join("/")}
                    </Link>
                    <div className="mt-1">
                      <CapBadges caps={m.capabilities} max={3} variant="muted" />
                    </div>
                  </td>
                  <td className="px-3 py-3.5 text-right font-mono text-[13px] font-semibold tabular-nums text-ink">
                    {m.aa?.intelligence != null ? m.aa.intelligence.toFixed(1) : "—"}
                  </td>
                  <td className="hidden px-3 py-3.5 text-right font-mono text-[13px] tabular-nums text-ink-2 sm:table-cell">
                    {m.throughput ? fmtThroughput(m.throughput) : "—"}
                  </td>
                  <td className="hidden px-3 py-3.5 text-right font-mono text-[13px] tabular-nums text-ink-2 sm:table-cell">
                    {fmtLatency(m.latency)}
                  </td>
                  <td className="px-3 py-3.5 text-right font-mono text-[13px] tabular-nums text-ink">{fmtPrice(m.price_input)}</td>
                  <td className="px-4 py-3.5 text-right font-mono text-[13px] tabular-nums text-ink-2">{fmtContext(m.context_length)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <Footer />
    </div>
  );
}
