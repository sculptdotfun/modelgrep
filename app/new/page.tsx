import type { Metadata } from "next";
import Link from "next/link";
import { getCatalog } from "@/lib/catalog";
import { Footer } from "@/components/Footer";
import { SiteHeader } from "@/components/SiteHeader";
import { CapBadges, OwnerAvatar } from "@/components/ui";
import { fmtContext, fmtPrice, modelOwner } from "@/lib/format";

export const revalidate = 3600;

export const metadata: Metadata = {
  title: "New LLM Models — Latest AI Model Releases, Tracked Live",
  description:
    "The newest AI models, tracked as they ship: release dates, intelligence benchmarks, pricing and context windows for the latest LLM releases from OpenAI, Anthropic, Google, Qwen, DeepSeek and more.",
  keywords: ["new LLM models", "latest AI models", "new AI model releases", "newest LLM", "AI model release tracker"],
  alternates: { canonical: "/new" },
  openGraph: {
    title: "New LLM Models — Latest Releases, Tracked Live",
    description: "The newest AI models with benchmarks and pricing, updated continuously.",
    url: "/new",
    type: "website",
  },
};

function fmtDate(ts: number): string {
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const d = new Date(ts * 1000);
  return `${months[d.getUTCMonth()]} ${d.getUTCDate()}, ${d.getUTCFullYear()}`;
}

export default async function NewModelsPage() {
  const models = await getCatalog();
  const recent = [...models]
    .filter((m) => m.created > 0)
    .sort((a, b) => b.created - a.created)
    .slice(0, 50);

  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: "Newest AI models",
    numberOfItems: recent.length,
    itemListElement: recent.slice(0, 25).map((m, i) => ({
      "@type": "ListItem",
      position: i + 1,
      name: m.name,
      url: `https://modelgrep.com/models/${m.id}`,
    })),
  };

  return (
    <div className="min-h-screen">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }} />
      <div className="mx-auto w-full max-w-[1200px] px-5 py-7">
        <SiteHeader />

        <h1 className="font-display mt-8 text-[30px] font-bold text-ink sm:text-[34px]">New models</h1>
        <p className="mt-2 max-w-2xl text-[15px] leading-relaxed text-ink-2">
          The latest AI model releases, tracked as they land — with benchmarks, pricing and specs as soon as
          they&apos;re available.
        </p>

        <ol className="mt-7 divide-y divide-line overflow-hidden rounded-lg border border-line bg-surface">
          {recent.map((m) => (
            <li key={m.id}>
              <Link href={`/models/${m.id}`} className="group flex items-center gap-3 px-4 py-4 transition-colors hover:bg-surface-2/60">
                <OwnerAvatar owner={modelOwner(m.id)} size={30} />
                <div className="min-w-0 flex-1">
                  <div className="truncate font-mono text-[13px] font-medium text-ink group-hover:text-brand-ink">{m.id}</div>
                  <div className="mt-1 flex items-center gap-2">
                    <span className="font-mono text-[10px] uppercase tracking-wider text-ink-3">{fmtDate(m.created)}</span>
                    <CapBadges caps={m.capabilities} max={3} variant="muted" />
                  </div>
                </div>
                <div className="shrink-0 text-right">
                  <div className="font-mono text-sm font-bold tabular-nums text-ink">
                    {m.aa?.intelligence != null ? m.aa.intelligence.toFixed(1) : "—"}
                  </div>
                  <div className="text-[10px] uppercase tracking-wide text-ink-3">Intel</div>
                </div>
                <div className="hidden w-24 shrink-0 text-right font-mono text-xs text-ink-2 sm:block">
                  {m.price_input != null ? `${fmtPrice(m.price_input)}/M` : "—"}
                  <div className="text-[10px] text-ink-3">{fmtContext(m.context_length)} ctx</div>
                </div>
              </Link>
            </li>
          ))}
        </ol>
      </div>
      <Footer />
    </div>
  );
}
