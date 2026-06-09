import Link from "next/link";

// Minimal inline page header — wordmark + quiet links, no sticky chrome.
export function SiteHeader() {
  return (
    <div className="flex items-center justify-between">
      <Link href="/" className="font-mono text-[15px] font-bold tracking-tight text-ink">
        model<span className="text-brand">grep</span>
      </Link>
      <nav className="flex items-center gap-4 text-[13px] text-ink-3">
        <Link href="/" className="transition-colors hover:text-ink">
          Leaderboard
        </Link>
        <Link href="/compare" className="hidden transition-colors hover:text-ink sm:block">
          Compare
        </Link>
        <Link href="/new" className="transition-colors hover:text-ink">
          New
        </Link>
        <Link href="/best/smartest" className="transition-colors hover:text-ink">
          Rankings
        </Link>
        <Link href="/blog" className="transition-colors hover:text-ink">
          Blog
        </Link>
      </nav>
    </div>
  );
}
