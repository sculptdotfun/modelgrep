import Link from "next/link";
import { BarChart3, BookOpen, GitCompareArrows, Sparkles, Trophy } from "lucide-react";

// Brand mark — a gradient prompt tile: chevron + match cursor. "grep, on a
// gradient." The unique gradient id is fine to repeat across instances.
export function Mark({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 100 100" aria-hidden>
      <defs>
        <linearGradient id="mgMark" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#7d5cff" />
          <stop offset="1" stopColor="#5a32f0" />
        </linearGradient>
      </defs>
      <rect width="100" height="100" rx="28" fill="url(#mgMark)" />
      <path d="M31 33 L51 50 L31 67" fill="none" stroke="#ffffff" strokeWidth="10" strokeLinecap="round" strokeLinejoin="round" />
      <rect x="55" y="58" width="24" height="10" rx="5" fill="#ffffff" />
    </svg>
  );
}

const NAV = [
  { href: "/", label: "Leaderboard", Icon: BarChart3 },
  { href: "/best", label: "Rankings", Icon: Trophy },
  { href: "/compare", label: "Compare", Icon: GitCompareArrows, hideSm: true },
  { href: "/new", label: "New", Icon: Sparkles },
  { href: "/blog", label: "Guides", Icon: BookOpen, hideSm: true },
];

// Global sticky navigation — rendered once in the root layout, glass over
// scrolling content with a hairline border.
export function SiteNav() {
  return (
    <header className="glass sticky top-0 z-50 border-b border-line">
      <div className="mx-auto flex h-14 w-full max-w-[1200px] items-center justify-between px-5">
        <Link href="/" className="flex items-center gap-2.5 text-[15px] font-bold tracking-tight text-ink">
          <Mark size={26} />
          <span className="font-display">
            model<span className="gradient-ink">grep</span>
          </span>
        </Link>
        <nav className="flex items-center gap-0.5">
          {NAV.map(({ href, label, Icon, hideSm }) => (
            <Link
              key={href}
              href={href}
              className={`flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-[13px] font-medium text-ink-2 transition-colors hover:bg-surface-2 hover:text-ink ${hideSm ? "hidden sm:flex" : "flex"}`}
            >
              <Icon className="size-[15px] text-ink-3" strokeWidth={2} />
              <span className="hidden md:inline">{label}</span>
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
