import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import clsx from "clsx";
import { getCatalog, getModel } from "@/lib/catalog";
import { COMPARE_METRICS, winner } from "@/lib/compare";
import { Footer } from "@/components/Footer";
import { SiteHeader } from "@/components/SiteHeader";
import { CapBadges, OwnerAvatar } from "@/components/ui";
import { modelOwner } from "@/lib/format";
import type { Model } from "@/lib/types";

export const revalidate = 3600;

type Params = { slug: string[] };

// Parse "/compare/<idA>/vs/<idB>" → ordered, canonicalized pair.
function parsePair(slug: string[]): { a: string; b: string } | null {
  const i = slug.indexOf("vs");
  if (i <= 0 || i >= slug.length - 1) return null;
  const a = slug.slice(0, i).join("/");
  const b = slug.slice(i + 1).join("/");
  if (!a || !b || a === b) return null;
  // canonical order: alphabetical, so A-vs-B and B-vs-A are one page
  return a < b ? { a, b } : { a: b, b: a };
}

export async function generateStaticParams() {
  const models = await getCatalog();
  const top = models
    .filter((m) => m.aa?.intelligence != null)
    .sort((x, y) => y.aa!.intelligence! - x.aa!.intelligence!)
    .slice(0, 12);
  const params: { slug: string[] }[] = [];
  for (let i = 0; i < top.length; i++) {
    for (let j = i + 1; j < top.length; j++) {
      const [a, b] = [top[i].id, top[j].id].sort();
      params.push({ slug: [...a.split("/"), "vs", ...b.split("/")] });
    }
  }
  return params;
}

export async function generateMetadata({ params }: { params: Promise<Params> }): Promise<Metadata> {
  const { slug } = await params;
  const pair = parsePair(slug);
  if (!pair) return { title: "Compare models" };
  const [a, b] = await Promise.all([getModel(pair.a), getModel(pair.b)]);
  if (!a || !b) return { title: "Compare models" };

  const canonical = `/compare/${pair.a}/vs/${pair.b}`;
  const title = `${a.name} vs ${b.name} — Benchmarks, Speed & Pricing`;
  const description = `${a.name} vs ${b.name}: side-by-side comparison of intelligence benchmarks, coding, speed, latency, context window and per-token pricing. See which model is smarter, faster, and cheaper.`;
  return {
    title,
    description,
    keywords: [`${a.name} vs ${b.name}`, `${b.name} vs ${a.name}`, `compare ${a.name}`, "LLM comparison"],
    alternates: { canonical },
    openGraph: { title, description, url: canonical, type: "article" },
    twitter: { card: "summary_large_image", title, description },
  };
}

function MetricRow({ metric, a, b }: { metric: (typeof COMPARE_METRICS)[number]; a: Model; b: Model }) {
  const w = winner(metric, a, b);
  return (
    <tr className="border-t border-line">
      <td className="px-4 py-3 text-sm text-ink-2">{metric.label}</td>
      <td className={clsx("px-4 py-3 text-right font-mono text-[13px] tabular-nums", w === -1 ? "font-bold text-elite" : "text-ink")}>
        {metric.fmt(metric.get(a))}
        {w === -1 && <span className="ml-1.5 text-[10px]">✓</span>}
      </td>
      <td className={clsx("px-4 py-3 text-right font-mono text-[13px] tabular-nums", w === 1 ? "font-bold text-elite" : "text-ink")}>
        {metric.fmt(metric.get(b))}
        {w === 1 && <span className="ml-1.5 text-[10px]">✓</span>}
      </td>
    </tr>
  );
}

function Head({ m }: { m: Model }) {
  return (
    <Link href={`/models/${m.id}`} className="group flex flex-col items-center gap-2 text-center">
      <OwnerAvatar owner={modelOwner(m.id)} size={48} />
      <span className="text-[15px] font-semibold text-ink group-hover:text-brand-ink">{m.name}</span>
      <span className="font-mono text-[11px] text-ink-3">{m.id}</span>
    </Link>
  );
}

export default async function ComparePage({ params }: { params: Promise<Params> }) {
  const { slug } = await params;
  const pair = parsePair(slug);
  if (!pair) notFound();
  const [a, b] = await Promise.all([getModel(pair.a), getModel(pair.b)]);
  if (!a || !b) notFound();

  // tally wins for the verdict
  let aw = 0;
  let bw = 0;
  for (const metric of COMPARE_METRICS) {
    const w = winner(metric, a, b);
    if (w === -1) aw++;
    else if (w === 1) bw++;
  }
  const lead = aw === bw ? null : aw > bw ? a : b;
  const verdict = lead
    ? `${lead.name} wins on more metrics (${Math.max(aw, bw)} of ${aw + bw}), but the right pick depends on what you optimize for — see the breakdown below.`
    : `${a.name} and ${b.name} are evenly matched; the right pick depends on your priorities.`;

  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: `${a.name} vs ${b.name}`,
    itemListElement: [a, b].map((m, i) => ({ "@type": "ListItem", position: i + 1, name: m.name, url: `https://modelgrep.com/models/${m.id}` })),
  };

  return (
    <div className="min-h-screen">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }} />
      <div className="mx-auto w-full max-w-[860px] px-5 py-7">
        <SiteHeader />

        <h1 className="font-display mt-6 text-center text-[28px] font-bold text-ink sm:text-[32px]">
          {a.name} <span className="text-ink-3">vs</span> {b.name}
        </h1>

        <div className="mt-6 grid grid-cols-[1fr_auto_1fr] items-center gap-4">
          <Head m={a} />
          <span className="text-sm font-semibold text-ink-3">VS</span>
          <Head m={b} />
        </div>

        <p className="mx-auto mt-6 max-w-xl text-center text-[15px] leading-relaxed text-ink-2">{verdict}</p>

        <div className="card-shadow mt-7 overflow-hidden rounded-xl border border-line bg-surface">
          <table className="w-full">
            <thead>
              <tr className="text-[10px] uppercase tracking-wider text-ink-3">
                <th className="px-4 py-2.5 text-left font-semibold">Metric</th>
                <th className="px-4 py-2.5 text-right font-semibold">{a.name}</th>
                <th className="px-4 py-2.5 text-right font-semibold">{b.name}</th>
              </tr>
            </thead>
            <tbody>
              {COMPARE_METRICS.map((metric) => (
                <MetricRow key={metric.key} metric={metric} a={a} b={b} />
              ))}
              <tr className="border-t border-line">
                <td className="px-4 py-3 text-sm text-ink-2">Capabilities</td>
                <td className="px-4 py-3 text-right">
                  <span className="inline-flex justify-end">
                    <CapBadges caps={a.capabilities} variant="muted" />
                  </span>
                </td>
                <td className="px-4 py-3 text-right">
                  <span className="inline-flex justify-end">
                    <CapBadges caps={b.capabilities} variant="muted" />
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="mt-5 flex justify-center gap-3 text-sm">
          <Link href={`/models/${a.id}`} className="rounded-lg border border-line bg-surface px-3.5 py-2 font-medium text-ink-2 hover:text-ink">
            {a.name} details →
          </Link>
          <Link href={`/models/${b.id}`} className="rounded-lg border border-line bg-surface px-3.5 py-2 font-medium text-ink-2 hover:text-ink">
            {b.name} details →
          </Link>
        </div>
      </div>
      <Footer />
    </div>
  );
}
