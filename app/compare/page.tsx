import type { Metadata } from "next";
import Link from "next/link";
import { getCatalog } from "@/lib/catalog";
import { Footer } from "@/components/Footer";
import { SiteHeader } from "@/components/SiteHeader";
import { OwnerAvatar } from "@/components/ui";
import { modelOwner } from "@/lib/format";

export const revalidate = 3600;

export const metadata: Metadata = {
  title: "Compare LLMs Side by Side — Benchmarks, Speed & Pricing",
  description:
    "Head-to-head AI model comparisons: intelligence benchmarks, coding scores, speed, latency, context windows and pricing. GPT vs Claude vs Gemini vs Qwen vs DeepSeek and every other matchup.",
  keywords: ["compare LLMs", "LLM comparison", "GPT vs Claude", "AI model comparison tool"],
  alternates: { canonical: "/compare" },
  openGraph: {
    title: "Compare LLMs side by side",
    description: "Head-to-head model comparisons across benchmarks, speed and price.",
    url: "/compare",
    type: "website",
  },
};

export default async function CompareIndex() {
  const models = await getCatalog();
  const top = models
    .filter((m) => m.aa?.intelligence != null)
    .sort((a, b) => b.aa!.intelligence! - a.aa!.intelligence!)
    .slice(0, 14);

  // Popular matchups: every pair among the top models, strongest pairs first.
  const pairs: { a: (typeof top)[number]; b: (typeof top)[number] }[] = [];
  for (let i = 0; i < top.length; i++) {
    for (let j = i + 1; j < top.length; j++) {
      pairs.push({ a: top[i], b: top[j] });
    }
  }

  return (
    <div className="min-h-screen">
      <div className="mx-auto w-full max-w-[1200px] px-5 py-7">
        <SiteHeader />

        <h1 className="font-display mt-8 text-[30px] font-bold text-ink sm:text-[34px]">Compare models</h1>
        <p className="mt-2 max-w-2xl text-[15px] leading-relaxed text-ink-2">
          Head-to-head comparisons across intelligence benchmarks, coding, design Elo, speed, latency, context and
          pricing. Pick any matchup below, or open any model page and choose an opponent.
        </p>

        <div className="mt-7 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          {pairs.slice(0, 45).map(({ a, b }) => {
            const [x, y] = [a.id, b.id].sort();
            return (
              <Link
                key={`${a.id}|${b.id}`}
                href={`/compare/${x}/vs/${y}`}
                className="card-lift group flex items-center gap-2.5 rounded-lg border border-line bg-surface px-3.5 py-2.5"
              >
                <span className="flex shrink-0 -space-x-1.5">
                  <OwnerAvatar owner={modelOwner(a.id)} size={22} />
                  <OwnerAvatar owner={modelOwner(b.id)} size={22} />
                </span>
                <span className="truncate text-[13px] font-medium text-ink group-hover:text-brand-ink">
                  {a.id.split("/")[1]} <span className="text-ink-3">vs</span> {b.id.split("/")[1]}
                </span>
              </Link>
            );
          })}
        </div>
      </div>
      <Footer />
    </div>
  );
}
