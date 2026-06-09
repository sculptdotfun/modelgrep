import Link from "next/link";
import { COLLECTIONS } from "@/lib/collections";
import { BLOG_POSTS } from "@/lib/blog";

export function Footer() {
  return (
    <footer className="mt-12 border-t border-line">
      <div className="mx-auto w-full max-w-[1320px] px-5 py-9">
        <div className="flex flex-col gap-6 sm:flex-row sm:justify-between">
          <div className="max-w-xs">
            <Link href="/" className="font-mono text-[15px] font-bold tracking-tight text-ink">
              model<span className="text-brand">grep</span>
            </Link>
            <p className="mt-2 text-[13px] text-ink-2">The leaderboard to find &amp; understand every LLM — ranked by benchmarks, speed and price.</p>
          </div>
          <div className="flex gap-12">
            <div>
              <div className="mb-2.5 text-[11px] font-semibold uppercase tracking-wider text-ink-3">Rankings</div>
              <div className="grid grid-cols-2 gap-x-8 gap-y-1.5">
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
          </div>
        </div>
        <div className="mt-8 border-t border-line pt-5 text-xs text-ink-3">
          Benchmarks: Artificial Analysis &amp; Design Arena · Live data via OpenRouter
        </div>
      </div>
    </footer>
  );
}
