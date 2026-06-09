import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { getCatalog } from "@/lib/catalog";
import { COLLECTIONS, getCollection } from "@/lib/collections";
import { Footer } from "@/components/Footer";
import { SiteHeader } from "@/components/SiteHeader";
import { CapBadges, OwnerAvatar } from "@/components/ui";
import { fmtContext, fmtPrice, modelOwner } from "@/lib/format";

export const revalidate = 3600;

type Params = { slug: string };

export async function generateStaticParams() {
  return COLLECTIONS.map((c) => ({ slug: c.slug }));
}

export async function generateMetadata({ params }: { params: Promise<Params> }): Promise<Metadata> {
  const { slug } = await params;
  const c = getCollection(slug);
  if (!c) return { title: "Not found" };
  const title = `${c.title} (2026) — Ranked & Benchmarked`;
  return {
    title,
    description: c.blurb,
    keywords: [c.title, `best LLMs ${c.slug}`, "LLM rankings", "AI model comparison"],
    alternates: { canonical: `/best/${c.slug}` },
    openGraph: { title, description: c.blurb, url: `/best/${c.slug}`, type: "article" },
    twitter: { card: "summary_large_image", title, description: c.blurb },
  };
}

export default async function CollectionPage({ params }: { params: Promise<Params> }) {
  const { slug } = await params;
  const c = getCollection(slug);
  if (!c) notFound();
  const models = await getCatalog();
  const ranked = models.filter(c.filter).sort(c.sort).slice(0, 25);

  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: c.title,
    description: c.blurb,
    numberOfItems: ranked.length,
    itemListElement: ranked.map((m, i) => ({ "@type": "ListItem", position: i + 1, name: m.name, url: `https://modelgrep.com/models/${m.id}` })),
  };

  return (
    <div className="min-h-screen">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }} />
      <div className="mx-auto w-full max-w-[1200px] px-5 py-7">
        <SiteHeader />

        <nav className="mt-5 text-xs text-ink-3">
          <Link href="/" className="hover:text-ink-2">
            models
          </Link>
          <span className="mx-1.5">/</span>
          <span className="text-ink">{c.title}</span>
        </nav>

        <h1 className="font-display mt-4 text-[30px] font-bold text-ink sm:text-[34px]">{c.title}</h1>
        <p className="mt-3 max-w-2xl text-[15px] leading-relaxed text-ink-2">{c.blurb}</p>

        <ol className="card-shadow mt-7 divide-y divide-line overflow-hidden rounded-lg border border-line bg-surface">
          {ranked.map((m, i) => (
            <li key={m.id}>
              <Link href={`/models/${m.id}`} className="group flex items-center gap-3 px-4 py-3 transition-colors hover:bg-surface-2/60">
                <span className="w-5 shrink-0 text-right font-mono text-xs tabular-nums text-ink-3">{i + 1}</span>
                <OwnerAvatar owner={modelOwner(m.id)} size={30} />
                <div className="min-w-0 flex-1">
                  <div className="truncate font-mono text-[13px] font-medium text-ink group-hover:text-brand-ink">{m.id}</div>
                  <div className="mt-1">
                    <CapBadges caps={m.capabilities} max={3} variant="muted" />
                  </div>
                </div>
                <div className="shrink-0 text-right">
                  <div className="font-mono text-sm font-bold text-ink">{c.display(m)}</div>
                  <div className="text-[10px] uppercase tracking-wide text-ink-3">{c.metricLabel}</div>
                </div>
                <div className="hidden w-20 shrink-0 text-right font-mono text-xs text-ink-2 sm:block">
                  {m.price_input != null ? `${fmtPrice(m.price_input)}/M` : "—"}
                  <div className="text-[10px] text-ink-3">{fmtContext(m.context_length)} ctx</div>
                </div>
              </Link>
            </li>
          ))}
        </ol>

        <div className="mt-7">
          <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-ink-3">More rankings</div>
          <div className="flex flex-wrap gap-2">
            {COLLECTIONS.filter((x) => x.slug !== c.slug).map((x) => (
              <Link key={x.slug} href={`/best/${x.slug}`} className="rounded-md border border-line bg-surface px-3 py-1.5 text-[12px] text-ink-2 hover:text-ink">
                {x.title}
              </Link>
            ))}
          </div>
        </div>
      </div>
      <Footer />
    </div>
  );
}
