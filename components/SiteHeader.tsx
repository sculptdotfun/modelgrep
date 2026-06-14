import Link from "next/link";

// Brand mark — the >_ prompt glyph.
export function Mark({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 100 100" aria-hidden>
      <rect width="100" height="100" rx="20" fill="#101014" />
      <path d="M28 32 L52 50 L28 68" fill="none" stroke="#ffffff" strokeWidth="9" strokeLinecap="square" />
      <rect x="56" y="61" width="20" height="9" fill="#5b3df5" />
    </svg>
  );
}

// Minimal inline page header — wordmark + quiet links, no sticky chrome.
export function SiteHeader() {
  return (
    <div className="flex items-center justify-between">
      <Link href="/" className="flex items-center gap-2 font-mono text-[15px] font-bold tracking-tight text-ink">
        <Mark />
        <span>
          model<span className="text-brand">grep</span>
        </span>
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
        <Link href="/best" className="transition-colors hover:text-ink">
          Rankings
        </Link>
        <Link href="/blog" className="transition-colors hover:text-ink">
          Blog
        </Link>
      </nav>
    </div>
  );
}
