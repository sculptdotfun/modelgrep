import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { getCatalog, getModel } from "@/lib/catalog";
import { AnswerBox } from "@/components/AnswerBox";
import { Footer } from "@/components/Footer";
import { SiteHeader } from "@/components/SiteHeader";
import { CapBadges, OwnerAvatar } from "@/components/ui";
import { fmtContext, fmtLatency, fmtMonth, fmtPrice, fmtThroughput, modelOwner } from "@/lib/format";
import type { Model } from "@/lib/types";

export const revalidate = 3600;

type Params = { slug: string[] };

function short(m: Model): string {
  return m.name.includes(": ") ? m.name.split(": ").slice(1).join(": ") : m.name;
}

export async function generateStaticParams() {
  const models = await getCatalog();
  return [...models]
    .filter((m) => m.aa?.intelligence != null)
    .sort((a, b) => b.aa!.intelligence! - a.aa!.intelligence!)
    .slice(0, 40)
    .map((m) => ({ slug: m.id.split("/") }));
}

// Compute alternative buckets relative to a base model.
function computeAlts(base: Model, catalog: Model[]) {
  const baseIntel = base.aa?.intelligence ?? null;
  const others = catalog.filter((m) => m.id !== base.id);

  // Nearest by intelligence = the closest drop-in substitutes.
  const nearest =
    baseIntel != null
      ? [...others]
          .filter((m) => m.aa?.intelligence != null)
          .sort((a, b) => Math.abs(a.aa!.intelligence! - baseIntel) - Math.abs(b.aa!.intelligence! - baseIntel))
      : others.filter((m) => modelOwner(m.id) === modelOwner(base.id));

  const within = (m: Model) => baseIntel == null || (m.aa?.intelligence != null && m.aa.intelligence >= baseIntel - 8);

  const cheaper =
    base.price_input != null
      ? others
          .filter((m) => m.price_input != null && m.price_input < base.price_input! && within(m))
          .sort((a, b) => (b.aa?.intelligence ?? -1) - (a.aa?.intelligence ?? -1))
          .slice(0, 6)
      : [];
  const faster = others
    .filter((m) => m.throughput > base.throughput && within(m))
    .sort((a, b) => b.throughput - a.throughput)
    .slice(0, 6);
  const smarter =
    baseIntel != null
      ? others
          .filter((m) => m.aa?.intelligence != null && m.aa.intelligence > baseIntel)
          .sort((a, b) => a.aa!.intelligence! - b.aa!.intelligence!)
          .slice(0, 6)
      : [];
  const openSource = others
    .filter((m) => m.hugging_face_id && within(m))
    .sort((a, b) => (b.aa?.intelligence ?? -1) - (a.aa?.intelligence ?? -1))
    .slice(0, 6);

  const closest = nearest.find((m) => modelOwner(m.id) !== modelOwner(base.id)) ?? nearest[0] ?? null;
  return { closest, cheaper, faster, smarter, openSource };
}

export async function generateMetadata({ params }: { params: Promise<Params> }): Promise<Metadata> {
  const { slug } = await params;
  const m = await getModel(slug.join("/"));
  if (!m) return { title: "Model not found" };
  const name = short(m);
  const title = `${name} Alternatives — Cheaper, Faster & Open-Source Models Like ${name}`;
  const description = `The best alternatives to ${name}: cheaper, faster, smarter and open-source models with comparable intelligence, ranked by benchmark, speed and price.`;
  return {
    title,
    description,
    keywords: [`${name} alternative`, `${name} alternatives`, `models like ${name}`, `cheaper than ${name}`, `${name} vs`],
    alternates: { canonical: `/alternatives/${m.id}` },
    openGraph: { title, description, url: `/alternatives/${m.id}`, type: "article" },
    twitter: { card: "summary_large_image", title, description },
  };
}

function AltRow({ m, reason }: { m: Model; reason: string }) {
  return (
    <Link href={`/models/${m.id}`} className="group flex items-center gap-3 px-4 py-3.5 transition-colors hover:bg-surface-2/60">
      <OwnerAvatar owner={modelOwner(m.id)} size={28} />
      <div className="min-w-0 flex-1">
        <div className="truncate font-mono text-[13px] font-semibold text-ink group-hover:text-brand-ink">{m.id.split("/").slice(1).join("/")}</div>
        <div className="mt-0.5 truncate text-[11px] text-ink-3">{reason}</div>
      </div>
      <CapBadges caps={m.capabilities} max={2} variant="muted" />
    </Link>
  );
}

function AltSection({ title, models, reason }: { title: string; models: Model[]; reason: (m: Model) => string }) {
  if (!models.length) return null;
  return (
    <section className="mt-7">
      <h2 className="mb-2.5 text-[11px] font-semibold uppercase tracking-wider text-ink-3">{title}</h2>
      <div className="card-shadow divide-y divide-line overflow-hidden rounded-lg border border-line bg-surface">
        {models.map((m) => (
          <AltRow key={m.id} m={m} reason={reason(m)} />
        ))}
      </div>
    </section>
  );
}

export default async function AlternativesPage({ params }: { params: Promise<Params> }) {
  const { slug } = await params;
  const id = slug.join("/");
  const [base, catalog] = await Promise.all([getModel(id), getCatalog()]);
  if (!base) notFound();

  const name = short(base);
  const { closest, cheaper, faster, smarter, openSource } = computeAlts(base, catalog);
  const updated = fmtMonth(new Date());

  const parts: string[] = [];
  if (closest) parts.push(`The closest alternative to ${name} is ${short(closest)} — comparable intelligence (${closest.aa?.intelligence?.toFixed(1) ?? "—"} vs ${base.aa?.intelligence?.toFixed(1) ?? "—"})`);
  if (cheaper[0]) parts.push(`for a cheaper option, ${short(cheaper[0])} runs ${fmtPrice(cheaper[0].price_input)}/M vs ${fmtPrice(base.price_input)}/M`);
  if (faster[0]) parts.push(`for more speed, ${short(faster[0])} hits ${fmtThroughput(faster[0].throughput)} t/s`);
  if (openSource[0]) parts.push(`and the best open-weight substitute is ${short(openSource[0])}`);
  const answer = parts.join("; ").replace(/;([^;]*)$/, ";$1") + ".";

  const faqs = [
    closest && { q: `What is the best alternative to ${name}?`, a: `${short(closest)} is the closest alternative to ${name}, with an Intelligence Index of ${closest.aa?.intelligence?.toFixed(1) ?? "—"} (${name} scores ${base.aa?.intelligence?.toFixed(1) ?? "—"}) at ${fmtPrice(closest.price_input)} per million input tokens.` },
    cheaper[0] && { q: `Is there a cheaper alternative to ${name}?`, a: `Yes — ${short(cheaper[0])} costs ${fmtPrice(cheaper[0].price_input)} per million input tokens versus ${fmtPrice(base.price_input)} for ${name}, while staying within ~8 points of its Intelligence Index.` },
    openSource[0] && { q: `Is there an open-source alternative to ${name}?`, a: `${short(openSource[0])} is the strongest open-weight alternative to ${name} — downloadable from Hugging Face and self-hostable, scoring ${openSource[0].aa?.intelligence?.toFixed(1) ?? "—"} on intelligence.` },
  ].filter(Boolean) as { q: string; a: string }[];

  const jsonLd = [
    {
      "@context": "https://schema.org",
      "@type": "CollectionPage",
      name: `${name} alternatives`,
      url: `https://modelgrep.com/alternatives/${base.id}`,
      description: answer,
      dateModified: new Date().toISOString(),
    },
    {
      "@context": "https://schema.org",
      "@type": "FAQPage",
      mainEntity: faqs.map((q) => ({ "@type": "Question", name: q.q, acceptedAnswer: { "@type": "Answer", text: q.a } })),
    },
  ];

  return (
    <div className="min-h-screen">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }} />
      <div className="mx-auto w-full max-w-[1200px] px-5 py-7">
        <SiteHeader />

        <nav className="mt-5 text-xs text-ink-3">
          <Link href={`/models/${base.id}`} className="hover:text-ink-2">{base.id.split("/").slice(1).join("/")}</Link>
          <span className="mx-1.5">/</span>
          <span className="text-ink">alternatives</span>
        </nav>

        <h1 className="font-display mt-4 text-[28px] font-bold leading-tight text-ink sm:text-[32px]">{name} alternatives</h1>
        <AnswerBox answer={answer} updated={updated} />

        <div className="grid gap-x-6 lg:grid-cols-2">
          <AltSection title="Closest match" models={closest ? [closest] : []} reason={(m) => `Intelligence ${m.aa?.intelligence?.toFixed(1) ?? "—"} · ${fmtPrice(m.price_input)}/M · ${fmtThroughput(m.throughput)} t/s`} />
          <AltSection title="Cheaper alternatives" models={cheaper} reason={(m) => `${fmtPrice(m.price_input)}/M input · ${m.aa?.intelligence?.toFixed(1) ?? "—"} intel`} />
          <AltSection title="Faster alternatives" models={faster} reason={(m) => `${fmtThroughput(m.throughput)} t/s · ${fmtLatency(m.latency)} ttft`} />
          <AltSection title="Smarter alternatives" models={smarter} reason={(m) => `${m.aa?.intelligence?.toFixed(1) ?? "—"} intel · ${fmtPrice(m.price_input)}/M`} />
          <AltSection title="Open-source alternatives" models={openSource} reason={(m) => `${m.aa?.intelligence?.toFixed(1) ?? "—"} intel · open weights · ${fmtContext(m.context_length)} ctx`} />
        </div>

        {faqs.length > 0 && (
          <section className="mt-9">
            <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-wider text-ink-3">Frequently asked</h2>
            <div className="card-shadow divide-y divide-line rounded-lg border border-line bg-surface">
              {faqs.map((q) => (
                <div key={q.q} className="px-4 py-3.5">
                  <h3 className="text-[15px] font-semibold text-ink">{q.q}</h3>
                  <p className="mt-1.5 max-w-3xl text-sm leading-relaxed text-ink-2">{q.a}</p>
                </div>
              ))}
            </div>
          </section>
        )}

        <div className="mt-8 flex flex-wrap gap-2">
          <Link href={`/models/${base.id}`} className="rounded-md border border-line bg-surface px-3.5 py-2 text-sm font-medium text-ink-2 hover:text-ink">← {name} details</Link>
          <Link href={`/makers/${modelOwner(base.id)}`} className="rounded-md border border-line bg-surface px-3.5 py-2 text-sm font-medium text-ink-2 hover:text-ink">All {modelOwner(base.id)} models</Link>
        </div>
      </div>
      <Footer />
    </div>
  );
}
