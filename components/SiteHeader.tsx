import Link from "next/link";
import { BarChart3, GitCompareArrows, Sparkles, Trophy, BookOpen } from "lucide-react";

// Brand mark — the >_ prompt glyph.
export function Mark({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 100 100" aria-hidden>
      <rect width="100" height="100" rx="24" fill="#0a0a0b" />
      <path d="M28 32 L52 50 L28 68" fill="none" stroke="#ffffff" strokeWidth="9" strokeLinecap="round" strokeLinejoin="round" />
      <rect x="56" y="61" width="20" height="9" rx="2" fill="#6c5cf7" />
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

// Inline page header — wordmark + quiet icon links.
export function SiteHeader() {
  return (
    <div className="flex items-center justify-between">
      <Link href="/" className="group flex items-center gap-2.5 text-[15px] font-bold tracking-tight text-ink">
        <Mark size={26} />
        <span className="font-display">
          model<span className="text-brand">grep</span>
        </span>
      </Link>
      <nav className="flex items-center gap-1">
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
  );
}
