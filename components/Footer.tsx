import Link from "next/link";
import { BookText, GitCompareArrows, Sparkles } from "lucide-react";
import { COLLECTIONS } from "@/lib/collections";
import { BLOG_POSTS } from "@/lib/blog";
import { Mark } from "./SiteNav";

export function Footer() {
  return (
    <footer className="mt-12 border-t border-line">
      <div className="mx-auto w-full max-w-[1200px] px-5 py-9">
        <div className="flex flex-col gap-6 sm:flex-row sm:justify-between">
          <div className="max-w-xs">
            <Link href="/" className="flex items-center gap-2 font-mono text-[15px] font-bold tracking-tight text-ink">
              <Mark size={16} />
              <span>
                model<span className="text-brand">grep</span>
              </span>
            </Link>
            <p className="mt-2 text-[13px] text-ink-2">The leaderboard to find &amp; understand every LLM — ranked by benchmarks, speed and price.</p>
          </div>
          <div className="flex gap-12">
            <div>
              <div className="mb-2.5 text-[11px] font-semibold uppercase tracking-wider text-ink-3">Rankings</div>
              <div className="grid grid-cols-2 gap-x-8 gap-y-1.5">
                <Link href="/best" className="text-[13px] font-medium text-ink hover:text-brand-ink">
                  All rankings
                </Link>
                <Link href="/best/small" className="text-[13px] text-ink-2 hover:text-brand-ink">
                  Small &amp; Fast LLMs
                </Link>
                {COLLECTIONS.map((c) => (
                  <Link key={c.slug} href={`/best/${c.slug}`} className="text-[13px] text-ink-2 hover:text-brand-ink">
                    {c.title}
                  </Link>
                ))}
              </div>
            </div>
            <div>
              <div className="mb-2.5 text-[11px] font-semibold uppercase tracking-wider text-ink-3">Guides</div>
              <div className="flex flex-col gap-1.5">
                <Link href="/blog" className="text-[13px] text-ink-2 hover:text-brand-ink">
                  All guides
                </Link>
                {BLOG_POSTS.slice(0, 4).map((p) => (
                  <Link key={p.slug} href={`/blog/${p.slug}`} className="max-w-[220px] truncate text-[13px] text-ink-2 hover:text-brand-ink">
                    {p.title}
                  </Link>
                ))}
              </div>
            </div>
            <div>
              <div className="mb-2.5 text-[11px] font-semibold uppercase tracking-wider text-ink-3">Browse</div>
              <div className="flex flex-col gap-1.5">
                <Link href="/new" className="flex items-center gap-1.5 text-[13px] text-ink-2 hover:text-brand-ink">
                  <Sparkles className="size-3.5 text-ink-3" strokeWidth={2} /> New models
                </Link>
                <Link href="/compare" className="flex items-center gap-1.5 text-[13px] text-ink-2 hover:text-brand-ink">
                  <GitCompareArrows className="size-3.5 text-ink-3" strokeWidth={2} /> Compare models
                </Link>
                <Link href="/glossary" className="flex items-center gap-1.5 text-[13px] text-ink-2 hover:text-brand-ink">
                  <BookText className="size-3.5 text-ink-3" strokeWidth={2} /> LLM glossary
                </Link>
                {["openai", "anthropic", "google", "qwen", "deepseek", "meta-llama"].map((o) => (
                  <Link key={o} href={`/makers/${o}`} className="text-[13px] text-ink-2 hover:text-brand-ink">
                    {o === "meta-llama" ? "Meta" : o.charAt(0).toUpperCase() + o.slice(1)} models
                  </Link>
                ))}
              </div>
            </div>
          </div>
        </div>
        <div className="mt-8 border-t border-line pt-5 text-xs text-ink-3">
          Benchmarks: Artificial Analysis &amp; Design Arena · Live data via OpenRouter
        </div>
      </div>
    </footer>
  );
}
