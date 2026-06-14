import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { getCatalog } from "@/lib/catalog";
import { COLLECTIONS } from "@/lib/collections";
import { resolveFacet, makerOptions, rowStats } from "@/lib/facets";
import { AnswerBox } from "@/components/AnswerBox";
import { Footer } from "@/components/Footer";
import { CapBadges, OwnerAvatar, RankPip } from "@/components/ui";
import { fmtContext, fmtMonth, fmtPrice, fmtThroughput, modelOwner } from "@/lib/format";

export const revalidate = 3600;

type Params = { slug: string[] };

// Pre-render the singles + the highest-value intersections (top makers × every
// ranking). The long-tail intersections render on demand and cache (ISR).
export async function generateStaticParams() {
  const catalog = await getCatalog();
  const bases = [...COLLECTIONS.map((c) => c.slug), "small"];
  const makers = makerOptions(catalog).slice(0, 10);
  const params: { slug: string[] }[] = [];
  for (const base of bases) {
    params.push({ slug: [base] });
    for (const mk of makers) params.push({ slug: [base, mk.slug] });
  }
  return params;
}

export async function generateMetadata({ params }: { params: Promise<Params> }): Promise<Metadata> {
  const { slug } = await params;
  const f = resolveFacet(slug, await getCatalog());
  if (!f) return { title: "Not found" };
  const desc = `${f.answer} ${f.blurb}`.slice(0, 300);
  const kw = f.maker
    ? [f.h1, `${f.maker.displayName} ${f.base.metricLabel.toLowerCase()}`, `best ${f.maker.displayName} model`, "LLM comparison"]
    : [f.h1, `best LLMs ${f.base.slug}`, "LLM rankings", "AI model comparison"];
  return {
    title: f.title,
    description: desc,
    keywords: kw,
    alternates: { canonical: f.canonical },
    openGraph: { title: f.title, description: desc, url: f.canonical, type: "article" },
    twitter: { card: "summary_large_image", title: f.title, description: desc },
  };
}

export default async function FacetPage({ params }: { params: Promise<Params> }) {
  const { slug } = await params;
  const catalog = await getCatalog();
  const f = resolveFacet(slug, catalog);
  if (!f) notFound();

  const { base, maker, models: ranked } = f;
  const updated = fmtMonth(new Date());
  const top = ranked[0];

  // Winner stat strip for the answer box (evidence density). Deduped by label so
  // the headline metric isn't repeated (e.g. Speed on a speed-ranked page).
  const stats: { label: string; value: string }[] = [];
  if (top) {
    const raw = [
      { label: base.metricLabel, value: base.display(top) },
      top.aa?.intelligence != null ? { label: "Intelligence", value: top.aa.intelligence.toFixed(1) } : null,
      top.throughput > 0 ? { label: "Speed", value: `${fmtThroughput(top.throughput)} t/s` } : null,
      top.price_input != null ? { label: "Input /M", value: fmtPrice(top.price_input) } : null,
      top.context_length ? { label: "Context", value: fmtContext(top.context_length) } : null,
    ].filter((s): s is { label: string; value: string } => s != null);
    const seen = new Set<string>();
    for (const s of raw) if (!seen.has(s.label)) (seen.add(s.label), stats.push(s));
  }

  const jsonLd = [
    {
      "@context": "https://schema.org",
      "@type": "CollectionPage",
      name: f.h1,
      url: `https://modelgrep.com${f.canonical}`,
      description: f.answer,
      dateModified: new Date().toISOString(),
      isPartOf: { "@type": "WebSite", name: "modelgrep", url: "https://modelgrep.com" },
    },
    {
      "@context": "https://schema.org",
      "@type": "ItemList",
      name: f.h1,
      numberOfItems: ranked.length,
      itemListElement: ranked.map((m, i) => ({
        "@type": "ListItem",
        position: i + 1,
        name: m.name,
        url: `https://modelgrep.com/models/${m.id}`,
      })),
    },
    {
      "@context": "https://schema.org",
      "@type": "BreadcrumbList",
      itemListElement: [
        { "@type": "ListItem", position: 1, name: "Rankings", item: "https://modelgrep.com/best" },
        ...(maker ? [{ "@type": "ListItem", position: 2, name: `${maker.displayName} models`, item: `https://modelgrep.com/makers/${maker.slug}` }] : []),
        { "@type": "ListItem", position: maker ? 3 : 2, name: f.h1, item: `https://modelgrep.com${f.canonical}` },
      ],
    },
    {
      "@context": "https://schema.org",
      "@type": "FAQPage",
      mainEntity: f.faqs.map((q) => ({ "@type": "Question", name: q.q, acceptedAnswer: { "@type": "Answer", text: q.a } })),
    },
  ];

  return (
    <div className="min-h-screen">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }} />
      <div className="mx-auto w-full max-w-[1200px] px-5 py-7">

        <nav className="mt-5 text-xs text-ink-3">
          <Link href="/best" className="hover:text-ink-2">rankings</Link>
          <span className="mx-1.5">/</span>
          {maker ? (
            <>
              <Link href={`/makers/${maker.slug}`} className="hover:text-ink-2">{maker.slug}</Link>
              <span className="mx-1.5">/</span>
              <span className="text-ink">{base.slug}</span>
            </>
          ) : (
            <span className="text-ink">{base.slug}</span>
          )}
        </nav>

        <h1 className="font-display mt-4 text-[30px] font-bold leading-tight text-ink sm:text-[34px]">{f.h1}</h1>

        <AnswerBox answer={f.answer} updated={updated} stats={stats} />

        <p className="mt-5 max-w-3xl text-[15px] leading-relaxed text-ink-2">{f.blurb}</p>

        <ol className="card-shadow mt-7 divide-y divide-line overflow-hidden rounded-lg border border-line bg-surface">
          {ranked.map((m, i) => {
            const extra = rowStats(base, m);
            return (
              <li key={m.id}>
                <Link href={`/models/${m.id}`} className="group flex items-center gap-3 px-4 py-4 transition-colors hover:bg-surface-2/60">
                  <span className="flex w-6 shrink-0 justify-center"><RankPip rank={i + 1} /></span>
                  <OwnerAvatar owner={modelOwner(m.id)} size={30} />
                  <div className="min-w-0 flex-1">
                    <div title={m.id} className="truncate font-mono text-[13px] font-semibold text-ink group-hover:text-brand-ink">
                      {m.id.split("/").slice(1).join("/")}
                    </div>
                    <div className="mt-1 flex flex-wrap items-center gap-x-2.5 gap-y-1">
                      <CapBadges caps={m.capabilities} max={3} variant="muted" />
                      {extra.length > 0 && <span className="font-mono text-[10px] text-ink-3">{extra.join(" · ")}</span>}
                    </div>
                  </div>
                  <div className="shrink-0 text-right">
                    <div className="font-mono text-sm font-bold text-ink">{base.display(m)}</div>
                    <div className="text-[10px] uppercase tracking-wide text-ink-3">{base.metricLabel}</div>
                  </div>
                </Link>
              </li>
            );
          })}
        </ol>

        {/* FAQ — stat-dense answers (FAQPage schema above) */}
        {f.faqs.length > 0 && (
          <section className="mt-9">
            <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-wider text-ink-3">Frequently asked</h2>
            <div className="card-shadow divide-y divide-line rounded-lg border border-line bg-surface">
              {f.faqs.map((q) => (
                <div key={q.q} className="px-4 py-3.5">
                  <h3 className="text-[15px] font-semibold text-ink">{q.q}</h3>
                  <p className="mt-1.5 max-w-3xl text-sm leading-relaxed text-ink-2">{q.a}</p>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Related intersections — the internal-link mesh */}
        {f.related.length > 0 && (
          <section className="mt-9">
            <h2 className="mb-2.5 text-[11px] font-semibold uppercase tracking-wider text-ink-3">
              {maker ? `More ${maker.displayName} rankings` : "By maker"}
            </h2>
            <div className="flex flex-wrap gap-2">
              {f.related.map((r) => (
                <Link key={r.href} href={r.href} className="rounded-md border border-line bg-surface px-3 py-1.5 text-[12px] capitalize text-ink-2 hover:text-brand-ink">
                  {r.label}
                </Link>
              ))}
            </div>
          </section>
        )}

        {/* Cross-link to the other base rankings */}
        <section className="mt-7">
          <h2 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-ink-3">All rankings</h2>
          <div className="flex flex-wrap gap-2">
            <Link href="/best/small" className="rounded-md border border-line bg-surface px-3 py-1.5 text-[12px] text-ink-2 hover:text-ink">Small &amp; Fast LLMs</Link>
            {COLLECTIONS.filter((x) => x.slug !== base.slug).map((x) => (
              <Link key={x.slug} href={`/best/${x.slug}`} className="rounded-md border border-line bg-surface px-3 py-1.5 text-[12px] text-ink-2 hover:text-ink">
                {x.title}
              </Link>
            ))}
          </div>
        </section>
      </div>
      <Footer />
    </div>
  );
}
