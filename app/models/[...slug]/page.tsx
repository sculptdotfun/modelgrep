import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import clsx from "clsx";
import { getCatalog, getModelDetail } from "@/lib/catalog";
import { Footer } from "@/components/Footer";
import { CapBadges, Metric, OwnerAvatar } from "@/components/ui";
import { ownerColor } from "@/lib/owners";
import {
  daCategoryLabel,
  eloTier,
  fmtContext,
  fmtCutoff,
  fmtLatency,
  fmtPrice,
  fmtThroughput,
  intelTier,
  modelOwner,
  pct,
  type Tier,
  tierColor,
  uptimeTier,
} from "@/lib/format";
import type { Model } from "@/lib/types";

export const revalidate = 3600;

type Params = { slug: string[] };

// Pre-render the most significant models at build time; the long tail renders
// on first request and is then cached (ISR). Keeps builds fast and resilient.
export async function generateStaticParams() {
  const models = await getCatalog();
  const top = [...models]
    .sort((a, b) => (b.aa?.intelligence ?? -1) - (a.aa?.intelligence ?? -1))
    .slice(0, 60);
  return top.map((m) => ({ slug: m.id.split("/") }));
}

export async function generateMetadata({ params }: { params: Promise<Params> }): Promise<Metadata> {
  const { slug } = await params;
  const id = slug.join("/");
  const m = await getModelDetail(id);
  if (!m) return { title: "Model not found" };

  const bits: string[] = [];
  if (m.aa?.intelligence != null) bits.push(`Intelligence ${m.aa.intelligence.toFixed(1)}`);
  if (m.throughput) bits.push(`${fmtThroughput(m.throughput)} t/s`);
  if (m.price_input != null) bits.push(`${fmtPrice(m.price_input)}/M in`);
  if (m.context_length) bits.push(`${fmtContext(m.context_length)} context`);
  const summary = bits.join(" · ");

  const title = `${m.name} — Benchmarks, Speed & Pricing`;
  const description = `${m.name}: ${summary || "specs, benchmarks and pricing"}. Compare intelligence, coding & design benchmarks, latency, context window, capabilities and per-provider pricing on modelgrep.`;
  const owner = modelOwner(m.id);

  return {
    title,
    description,
    keywords: [
      m.name,
      `${m.name} pricing`,
      `${m.name} benchmark`,
      `${m.name} api`,
      `${m.name} context window`,
      `${m.name} vs`,
      `${owner} models`,
      "LLM comparison",
    ],
    alternates: { canonical: `/models/${id}` },
    openGraph: {
      title,
      description,
      url: `/models/${id}`,
      type: "article",
      images: [{ url: `/og?id=${encodeURIComponent(id)}`, width: 1200, height: 630 }],
    },
    twitter: { card: "summary_large_image", title, description, images: [`/og?id=${encodeURIComponent(id)}`] },
  };
}

// rank helper: position of `id` in a sorted list, plus the total.
function rankOf(list: Model[], id: string): { rank: number; total: number } | null {
  const i = list.findIndex((m) => m.id === id);
  return i < 0 ? null : { rank: i + 1, total: list.length };
}

function ordinal(n: number): string {
  const s = ["th", "st", "nd", "rd"];
  const v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
}

function Section({ title, children, aside }: { title: string; children: React.ReactNode; aside?: React.ReactNode }) {
  return (
    <section className="mt-9">
      <div className="mb-3 flex items-baseline justify-between">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-ink-3">{title}</h2>
        {aside}
      </div>
      {children}
    </section>
  );
}

function Field({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between border-b border-line py-2.5 text-sm last:border-0">
      <span className="text-ink-2">{label}</span>
      <span className="font-mono text-ink">{value}</span>
    </div>
  );
}

// KPI card with tier-colored value + contextual rank/percentile sub.
function Kpi({ label, value, tier, sub }: { label: string; value: string; tier?: Tier; sub?: string }) {
  return (
    <div className="card-shadow rounded-xl border border-line bg-surface p-3.5">
      <div className="text-[10px] font-semibold uppercase tracking-wider text-ink-3">{label}</div>
      <div className={clsx("mt-1 text-[24px] font-bold leading-none tracking-tight", tier ? tierColor[tier] : "text-ink")}>{value}</div>
      {sub && <div className="mt-1.5 text-[11px] text-ink-3">{sub}</div>}
    </div>
  );
}

const BAR_FILL: Record<Tier, string> = { elite: "bg-elite", high: "bg-high", mid: "bg-mid", low: "bg-low", na: "bg-ink-3" };

// "Better than X% of models" stat with a bar.
function CompareBar({ label, pct, tier }: { label: string; pct: number; tier: Tier }) {
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <span className="text-sm text-ink-2">{label}</span>
        <span className={clsx("font-mono text-sm font-bold tabular-nums", tierColor[tier])}>{pct}%</span>
      </div>
      <div className="mt-1.5 h-2 overflow-hidden rounded-full bg-surface-2">
        <div className={clsx("h-full rounded-full", BAR_FILL[tier])} style={{ width: `${Math.max(2, pct)}%` }} />
      </div>
      <div className="mt-1 text-[11px] text-ink-3">of all ranked models</div>
    </div>
  );
}

// Benchmark row with a proportional bar.
function BenchBar({ label, value, frac, tier, display }: { label: string; value: number; frac: number; tier: Tier; display: string }) {
  return (
    <div className="flex items-center gap-3 border-b border-line py-2.5 last:border-0">
      <span className="w-40 shrink-0 text-sm text-ink-2">{label}</span>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-2">
        <div className={clsx("h-full rounded-full", BAR_FILL[tier])} style={{ width: `${Math.max(2, Math.min(100, frac * 100))}%` }} />
      </div>
      <span className={clsx("w-12 text-right font-mono text-[13px] font-semibold tabular-nums", tierColor[tier])}>{display}</span>
    </div>
  );
}

export default async function ModelPage({ params }: { params: Promise<Params> }) {
  const { slug } = await params;
  const id = slug.join("/");
  const [m, catalog] = await Promise.all([getModelDetail(id), getCatalog()]);
  if (!m) notFound();

  const aa = m.aa;
  const da = m.da;

  // Rankings across the catalog give the numbers meaning.
  const byIntel = catalog.filter((x) => x.aa?.intelligence != null).sort((a, b) => b.aa!.intelligence! - a.aa!.intelligence!);
  const bySpeed = catalog.filter((x) => x.throughput > 0).sort((a, b) => b.throughput - a.throughput);
  const byPrice = catalog.filter((x) => (x.price_input ?? 0) > 0).sort((a, b) => a.price_input! - b.price_input!);
  const intelRank = rankOf(byIntel, m.id);
  const speedRank = rankOf(bySpeed, m.id);
  const priceRank = rankOf(byPrice, m.id);

  // percentiles ("better than X% of the field")
  const intelPctl = intelRank ? Math.round(((intelRank.total - intelRank.rank) / intelRank.total) * 100) : null;
  const speedPctl = speedRank ? Math.round(((speedRank.total - speedRank.rank) / speedRank.total) * 100) : null;
  const cheaperPctl = priceRank ? Math.round(((priceRank.total - priceRank.rank) / priceRank.total) * 100) : null;

  // similar models — nearest by intelligence (or same owner as fallback)
  const baseIntel = aa?.intelligence ?? null;
  const similar =
    baseIntel != null
      ? catalog
          .filter((x) => x.id !== m.id && x.aa?.intelligence != null)
          .sort((a, b) => Math.abs(a.aa!.intelligence! - baseIntel) - Math.abs(b.aa!.intelligence! - baseIntel))
          .slice(0, 6)
      : catalog.filter((x) => x.id !== m.id && modelOwner(x.id) === modelOwner(m.id)).slice(0, 6);

  const url = `https://modelgrep.com/models/${m.id}`;
  const owner = modelOwner(m.id);

  // FAQ derived from the model's data — adds indexable long-tail content and
  // FAQ rich-result eligibility.
  const faqs: { q: string; a: string }[] = [];
  if (m.price_input != null) {
    faqs.push({
      q: `How much does ${m.name} cost?`,
      a: `${m.name} costs ${m.price_input === 0 ? "nothing (free tier)" : `${fmtPrice(m.price_input)} per million input tokens`}${m.price_output != null && m.price_input > 0 ? ` and ${fmtPrice(m.price_output)} per million output tokens` : ""} via OpenRouter${priceRank ? `, making it ${ordinal(priceRank.rank)} cheapest of ${priceRank.total} paid models` : ""}.`,
    });
  }
  if (aa?.intelligence != null) {
    faqs.push({
      q: `How smart is ${m.name}?`,
      a: `${m.name} scores ${aa.intelligence.toFixed(1)} on the Artificial Analysis Intelligence Index${intelRank ? `, ranking ${ordinal(intelRank.rank)} of ${intelRank.total} benchmarked models` : ""}${aa.gpqa != null ? `, with a GPQA Diamond score of ${pct(aa.gpqa, 0)}` : ""}.`,
    });
  }
  if (m.throughput || m.latency != null) {
    faqs.push({
      q: `How fast is ${m.name}?`,
      a: `${m.name} generates around ${m.throughput ? `${fmtThroughput(m.throughput)} tokens per second` : "—"}${m.latency != null ? ` with ${fmtLatency(m.latency)} time-to-first-token (p50)` : ""}${speedRank ? `, the ${ordinal(speedRank.rank)} fastest tracked model` : ""}.`,
    });
  }
  faqs.push({
    q: `What is ${m.name}'s context window?`,
    a: `${m.name} supports a ${fmtContext(m.context_length)}-token context window${m.max_output ? ` and can output up to ${fmtContext(m.max_output)} tokens` : ""}. It accepts ${m.input_modalities.join(", ")} input.`,
  });

  const jsonLd = [
    {
      "@context": "https://schema.org",
      "@type": "SoftwareApplication",
      name: m.name,
      url,
      applicationCategory: "Large Language Model",
      operatingSystem: "API",
      author: { "@type": "Organization", name: owner },
      offers: { "@type": "Offer", price: m.price_input ?? 0, priceCurrency: "USD", description: "Price per 1M input tokens" },
      ...(aa?.intelligence != null && {
        aggregateRating: { "@type": "AggregateRating", ratingValue: aa.intelligence, bestRating: 100, worstRating: 0, ratingCount: 1 },
      }),
    },
    {
      "@context": "https://schema.org",
      "@type": "BreadcrumbList",
      itemListElement: [
        { "@type": "ListItem", position: 1, name: "Models", item: "https://modelgrep.com/" },
        { "@type": "ListItem", position: 2, name: m.name, item: url },
      ],
    },
    {
      "@context": "https://schema.org",
      "@type": "FAQPage",
      mainEntity: faqs.map((f) => ({
        "@type": "Question",
        name: f.q,
        acceptedAnswer: { "@type": "Answer", text: f.a },
      })),
    },
  ];

  const aaEvals: { label: string; value: number | null; frac?: boolean; scale?: number }[] = [
    { label: "Intelligence Index", value: aa?.intelligence ?? null, scale: 65 },
    { label: "Coding Index", value: aa?.coding ?? null, scale: 65 },
    { label: "Agentic Index", value: aa?.agentic ?? null, scale: 65 },
    { label: "GPQA Diamond", value: aa?.gpqa ?? null, frac: true },
    { label: "Humanity's Last Exam", value: aa?.hle ?? null, frac: true },
    { label: "SciCode", value: aa?.scicode ?? null, frac: true },
    { label: "Tau²-Bench (agentic)", value: aa?.tau2 ?? null, frac: true },
  ];

  return (
    <div className="min-h-screen">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }} />

      <div className="mx-auto w-full max-w-[1320px] px-5 py-7">
        <Link href="/" className="font-mono text-[15px] font-bold tracking-tight text-ink">
          model<span className="text-brand">grep</span>
        </Link>
        <nav className="mb-5 mt-6 text-xs text-ink-3">
          <Link href="/" className="hover:text-ink-2">
            models
          </Link>
          <span className="mx-1.5">/</span>
          <span className="text-ink-2">{modelOwner(m.id)}</span>
          <span className="mx-1.5">/</span>
          <span className="text-ink">{m.id.split("/").slice(1).join("/")}</span>
        </nav>

        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="flex items-start gap-4">
            <OwnerAvatar owner={modelOwner(m.id)} size={52} />
            <div>
              <h1 className="text-[30px] font-bold leading-[1.1] tracking-tight text-ink">{m.name}</h1>
              <p className="mt-1 font-mono text-[13px] text-ink-3">{m.id}</p>
              <div className="mt-3 flex flex-wrap items-center gap-1.5">
                {intelRank && (
                  <span className="rounded-md bg-emerald-50 px-2 py-1 text-[11px] font-medium text-emerald-700 ring-1 ring-inset ring-emerald-200">
                    {ordinal(intelRank.rank)} smartest of {intelRank.total}
                  </span>
                )}
                {priceRank && priceRank.rank <= Math.ceil(priceRank.total * 0.25) && (
                  <span className="rounded-md bg-violet-50 px-2 py-1 text-[11px] font-medium text-violet-700 ring-1 ring-inset ring-violet-200">
                    Cheaper than {Math.round((1 - priceRank.rank / priceRank.total) * 100)}% of paid
                  </span>
                )}
                <span className="ml-0.5">
                  <CapBadges caps={m.capabilities} variant="muted" />
                </span>
              </div>
            </div>
          </div>
          <a
            href={`https://openrouter.ai/${m.id}`}
            target="_blank"
            rel="noopener noreferrer"
            className="shrink-0 rounded-lg bg-ink px-3.5 py-2 text-sm font-medium text-white transition-opacity hover:opacity-90"
          >
            Use via OpenRouter ↗
          </a>
        </div>

        {/* KPI cards with rank context */}
        <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
          <Kpi
            label="Intelligence"
            value={aa?.intelligence != null ? aa.intelligence.toFixed(1) : "—"}
            tier={intelTier(aa?.intelligence)}
            sub={intelRank ? `${ordinal(intelRank.rank)} of ${intelRank.total}` : aa?.intelligence_pct != null ? `${aa.intelligence_pct}th pct` : undefined}
          />
          <Kpi label="Design Elo" value={da?.elo != null ? String(da.elo) : "—"} sub={da?.category ? daCategoryLabel[da.category] ?? da.category : undefined} />
          <Kpi
            label="Speed"
            value={m.throughput ? fmtThroughput(m.throughput) : "—"}
            sub={speedRank ? `${ordinal(speedRank.rank)} fastest` : "tokens/sec"}
          />
          <Kpi label="Latency" value={fmtLatency(m.latency)} sub="first token" />
          <Kpi label="Input price" value={m.price_input != null ? fmtPrice(m.price_input) : "—"} sub={priceRank ? `${ordinal(priceRank.rank)} cheapest` : "/M tokens"} />
          <Kpi label="Context" value={fmtContext(m.context_length)} sub={m.max_output ? `${fmtContext(m.max_output)} max out` : undefined} />
        </div>

        {(intelPctl != null || speedPctl != null || cheaperPctl != null) && (
          <Section title="How it compares">
            <div className="card-shadow grid gap-x-8 gap-y-3.5 rounded-xl border border-line bg-surface p-4 sm:grid-cols-3">
              {intelPctl != null && <CompareBar label="Smarter than" pct={intelPctl} tier="elite" />}
              {speedPctl != null && <CompareBar label="Faster than" pct={speedPctl} tier="high" />}
              {cheaperPctl != null && <CompareBar label="Cheaper than" pct={cheaperPctl} tier="mid" />}
            </div>
          </Section>
        )}

        {m.description && (
          <Section title="Overview">
            <p className="max-w-3xl text-[15px] leading-relaxed text-ink-2">{m.description}</p>
          </Section>
        )}

        {(aa || da) && (
          <Section title="Benchmarks" aside={<span className="text-[11px] text-ink-3">independent · via OpenRouter</span>}>
            <div className="grid gap-4 lg:grid-cols-2">
              {aa && (
                <div className="card-shadow rounded-xl border border-line bg-surface p-4">
                  <div className="mb-3 flex items-center justify-between">
                    <span className="text-sm font-semibold text-ink">Artificial Analysis</span>
                    {aa.intelligence_pct != null && <span className="text-xs text-ink-3">{aa.intelligence_pct}th percentile</span>}
                  </div>
                  {aaEvals
                    .filter((e) => e.value != null)
                    .map((e) => {
                      const v = e.value!;
                      const frac = e.frac ? v : v / (e.scale ?? 65);
                      const tier = e.frac ? (v >= 0.7 ? "elite" : v >= 0.45 ? "high" : v >= 0.25 ? "mid" : "low") : intelTier(v);
                      return <BenchBar key={e.label} label={e.label} value={v} frac={frac} tier={tier} display={e.frac ? pct(v, 0) : v.toFixed(1)} />;
                    })}
                </div>
              )}
              {da && (
                <div className="card-shadow rounded-xl border border-line bg-surface p-4">
                  <div className="mb-3 flex items-center justify-between">
                    <span className="text-sm font-semibold text-ink">Design Arena · Elo</span>
                    {da.tournaments != null && <span className="text-xs text-ink-3">{da.tournaments.toLocaleString()} tournaments</span>}
                  </div>
                  {Object.entries(da.categories)
                    .sort((a, b) => (b[1].elo ?? 0) - (a[1].elo ?? 0))
                    .map(([cat, v]) => (
                      <BenchBar
                        key={cat}
                        label={daCategoryLabel[cat] ?? cat}
                        value={v.elo ?? 0}
                        frac={v.elo != null ? (v.elo - 700) / 700 : 0}
                        tier={eloTier(v.elo)}
                        display={String(v.elo ?? "—")}
                      />
                    ))}
                </div>
              )}
            </div>
          </Section>
        )}

        {m.provider_details.length > 0 && (
          <Section title={`Providers & pricing (${m.provider_details.length})`}>
            <div className="card-shadow overflow-hidden rounded-xl border border-line bg-surface">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-line text-[10px] uppercase tracking-wider text-ink-3">
                    <th className="px-4 py-2.5 text-left font-semibold">Provider</th>
                    <th className="px-3 py-2.5 text-right font-semibold">In $/M</th>
                    <th className="px-3 py-2.5 text-right font-semibold">Out $/M</th>
                    <th className="hidden px-3 py-2.5 text-right font-semibold sm:table-cell">Context</th>
                    <th className="px-4 py-2.5 text-right font-semibold">Uptime</th>
                  </tr>
                </thead>
                <tbody>
                  {m.provider_details.map((p, i) => (
                    <tr key={`${p.name}-${i}`} className="border-b border-line last:border-0 hover:bg-surface-2/60">
                      <td className="px-4 py-2.5">
                        <span className="font-medium text-ink">{p.name}</span>
                        {p.quantization && p.quantization !== "unknown" && (
                          <span className="ml-2 rounded bg-amber-50 px-1.5 py-0.5 text-[10px] text-amber-700 ring-1 ring-inset ring-amber-200">{p.quantization}</span>
                        )}
                        {p.caching && (
                          <span className="ml-1 rounded bg-emerald-50 px-1.5 py-0.5 text-[10px] text-emerald-700 ring-1 ring-inset ring-emerald-200">cache</span>
                        )}
                      </td>
                      <td className="px-3 py-2.5 text-right font-mono text-xs text-ink">{fmtPrice(p.price_input)}</td>
                      <td className="px-3 py-2.5 text-right font-mono text-xs text-ink">{fmtPrice(p.price_output)}</td>
                      <td className="hidden px-3 py-2.5 text-right font-mono text-xs text-ink-2 sm:table-cell">{fmtContext(p.context_length)}</td>
                      <td className="px-4 py-2.5 text-right">
                        <Metric value={p.uptime != null ? `${p.uptime}%` : "—"} tier={uptimeTier(p.uptime)} className="text-xs" />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        )}

        <Section title="Specifications">
          <div className="card-shadow grid gap-x-10 rounded-xl border border-line bg-surface px-4 py-1 sm:grid-cols-2">
            <Field label="Context window" value={fmtContext(m.context_length)} />
            <Field label="Max output" value={m.max_output ? fmtContext(m.max_output) : "—"} />
            <Field label="Knowledge cutoff" value={fmtCutoff(m.knowledge_cutoff)} />
            <Field label="Input modalities" value={m.input_modalities.join(", ")} />
            <Field label="Output modalities" value={m.output_modalities.join(", ")} />
            <Field label="Prompt caching" value={m.supports_caching ? "Supported" : "—"} />
            <Field label="Cache read price" value={m.price_cache_read != null ? `${fmtPrice(m.price_cache_read)}/M` : "—"} />
            <Field label="Moderated" value={m.is_moderated == null ? "—" : m.is_moderated ? "Yes" : "No"} />
            {m.hugging_face_id && (
              <Field
                label="Open weights"
                value={
                  <a href={`https://huggingface.co/${m.hugging_face_id}`} target="_blank" rel="noopener noreferrer" className="text-brand-ink hover:underline">
                    {m.hugging_face_id} ↗
                  </a>
                }
              />
            )}
          </div>
        </Section>

        {faqs.length > 0 && (
          <Section title={`${m.name} FAQ`}>
            <div className="card-shadow divide-y divide-line rounded-xl border border-line bg-surface">
              {faqs.map((f) => (
                <div key={f.q} className="px-4 py-3.5">
                  <h3 className="text-[15px] font-semibold text-ink">{f.q}</h3>
                  <p className="mt-1.5 max-w-3xl text-sm leading-relaxed text-ink-2">{f.a}</p>
                </div>
              ))}
            </div>
          </Section>
        )}

        {similar.length > 0 && (
          <Section title={baseIntel != null ? "Similar models" : `More from ${modelOwner(m.id)}`}>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {similar.map((s) => {
                const soc = ownerColor(modelOwner(s.id));
                return (
                  <Link
                    key={s.id}
                    href={`/models/${s.id}`}
                    className="card-shadow group rounded-xl border border-line bg-surface p-3.5 transition-colors hover:border-line-strong"
                  >
                    <div className="flex items-center gap-2">
                      <span className="size-2 shrink-0 rounded-full" style={{ background: soc }} />
                      <span className="truncate font-mono text-[13px] font-medium text-ink group-hover:text-brand-ink">{s.id}</span>
                    </div>
                    <div className="mt-2.5 flex items-center justify-between text-xs">
                      <span className="text-ink-3">
                        Intel{" "}
                        <span className={clsx("font-mono font-semibold", tierColor[intelTier(s.aa?.intelligence)])}>
                          {s.aa?.intelligence != null ? s.aa.intelligence.toFixed(1) : "—"}
                        </span>
                      </span>
                      <span className="font-mono text-ink-2">{fmtPrice(s.price_input)}/M</span>
                    </div>
                  </Link>
                );
              })}
            </div>
          </Section>
        )}
      </div>

      <Footer />
    </div>
  );
}
