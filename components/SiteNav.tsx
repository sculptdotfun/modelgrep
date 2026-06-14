import Link from "next/link";
import { BarChart3, BookOpen, GitCompareArrows, Sparkles, Trophy } from "lucide-react";

// Brand mark — a sharp monochrome prompt glyph. Restraint over decoration.
export function Mark({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 100 100" aria-hidden>
      <rect width="100" height="100" rx="22" fill="#0a0a0a" />
      <path d="M35 33 L55 50 L35 67" fill="none" stroke="#ffffff" strokeWidth="9" strokeLinecap="round" strokeLinejoin="round" />
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
