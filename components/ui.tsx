import clsx from "clsx";
import type { Capabilities } from "@/lib/types";
import { type Tier, tierColor } from "@/lib/format";
import { ownerColor } from "@/lib/owners";

// ---- rank pip — clean numerals, top three carry weight through type ----------

export function RankPip({ rank }: { rank: number }) {
  return (
    <span className={clsx("font-mono text-[13px] tabular-nums", rank <= 3 ? "font-bold text-ink" : "text-ink-3")}>
      {rank}
    </span>
  );
}

// ---- owner avatar (brand-colored initial) -----------------------------------

export function OwnerAvatar({ owner, size = 44 }: { owner: string; size?: number }) {
  const c = ownerColor(owner);
  return (
    <span
      className="inline-flex shrink-0 items-center justify-center rounded-lg font-bold text-white"
      style={{ width: size, height: size, fontSize: size * 0.42, background: c }}
    >
      {owner.charAt(0).toUpperCase()}
    </span>
  );
}

// ---- capability chips (restrained, soft-tinted) ------------------------------

const CAP_META: Record<string, { label: string; title: string; cls: string }> = {
  reasoning: { label: "Reasoning", title: "Extended reasoning / thinking", cls: "bg-violet-50 text-violet-700 ring-violet-200" },
  tools: { label: "Tools", title: "Function / tool calling", cls: "bg-blue-50 text-blue-700 ring-blue-200" },
  structured: { label: "JSON", title: "Structured outputs / response_format", cls: "bg-emerald-50 text-emerald-700 ring-emerald-200" },
  vision: { label: "Vision", title: "Accepts image input", cls: "bg-amber-50 text-amber-700 ring-amber-200" },
  audio_in: { label: "Audio", title: "Accepts audio input", cls: "bg-rose-50 text-rose-700 ring-rose-200" },
  image_out: { label: "Image out", title: "Generates images", cls: "bg-fuchsia-50 text-fuchsia-700 ring-fuchsia-200" },
};

const CAP_ORDER = ["reasoning", "tools", "structured", "vision", "audio_in", "image_out"] as const;

export function CapBadges({
  caps,
  max,
  variant = "color",
}: {
  caps: Capabilities;
  max?: number;
  variant?: "color" | "muted";
}) {
  const active = CAP_ORDER.filter((k) => caps[k]);
  const shown = max ? active.slice(0, max) : active;
  const extra = active.length - shown.length;
  return (
    <span className="inline-flex flex-wrap items-center gap-1">
      {shown.map((k) => (
        <span
          key={k}
          title={CAP_META[k].title}
          className={clsx(
            "rounded px-1.5 py-0.5 text-[10px] font-medium ring-1 ring-inset",
            variant === "muted" ? "bg-surface-2 text-ink-3 ring-line" : CAP_META[k].cls,
          )}
        >
          {CAP_META[k].label}
        </span>
      ))}
      {extra > 0 && <span className="text-[10px] text-ink-3">+{extra}</span>}
    </span>
  );
}

// ---- compact capability icons (for dense table rows) ------------------------

const CAP_ICON: Record<string, { title: string; path: React.ReactNode }> = {
  reasoning: { title: "Reasoning", path: <path d="M9.5 2a4.5 4.5 0 0 0-3 7.9c.5.4.5 1 .5 1.6v.5h4v-.5c0-.6 0-1.2.5-1.6A4.5 4.5 0 0 0 9.5 2ZM7.5 14h4M8 16h3" /> },
  tools: { title: "Tool calling", path: <path d="M11.5 4.5a2.5 2.5 0 0 0-3.3 3.3l-4.4 4.4 1.5 1.5 4.4-4.4a2.5 2.5 0 0 0 3.3-3.3l-1.4 1.4-1.1-1.1 1.4-1.4Z" /> },
  structured: { title: "Structured (JSON)", path: <path d="M6 3.5C4.5 3.5 4.5 5 4.5 6.5S4.5 8.5 3 8.5c1.5 0 1.5 1.5 1.5 3s0 1.5 1.5 1.5M11 3.5c1.5 0 1.5 1.5 1.5 3s0 2 1.5 2c-1.5 0-1.5 1.5-1.5 3s0 1.5-1.5 1.5" /> },
  vision: { title: "Vision", path: <><path d="M1.5 9S4 4.5 9 4.5 16.5 9 16.5 9 14 13.5 9 13.5 1.5 9 1.5 9Z" /><circle cx="9" cy="9" r="2" /></> },
};

const CAP_ICON_ORDER = ["reasoning", "tools", "structured", "vision"] as const;

export function CapIcons({ caps }: { caps: Capabilities }) {
  const active = CAP_ICON_ORDER.filter((k) => caps[k]);
  if (!active.length) return null;
  return (
    <span className="inline-flex items-center gap-1.5 text-ink-3">
      {active.map((k) => (
        <svg key={k} viewBox="0 0 18 18" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
          <title>{CAP_ICON[k].title}</title>
          {CAP_ICON[k].path}
        </svg>
      ))}
    </span>
  );
}

// ---- metric value (tier-colored, monospace) ----------------------------------

export function Metric({ value, tier, className }: { value: string; tier: Tier; className?: string }) {
  return <span className={clsx("font-mono tabular-nums", tierColor[tier], className)}>{value}</span>;
}

// ---- score pill with percentile bar -----------------------------------------

const BAR_BG: Record<Tier, string> = {
  elite: "bg-elite",
  high: "bg-high",
  mid: "bg-mid",
  low: "bg-low",
  na: "bg-ink-3",
};

export function ScorePill({ value, pctile, tier }: { value: string; pctile?: number | null; tier: Tier }) {
  return (
    <span className="inline-flex flex-col items-end gap-1">
      <span className={clsx("font-mono text-sm font-semibold tabular-nums", tierColor[tier])}>{value}</span>
      {pctile != null && (
        <span className="h-1 w-12 overflow-hidden rounded-[1px] bg-surface-2">
          <span className={clsx("block h-full rounded-[1px]", BAR_BG[tier])} style={{ width: `${Math.max(4, pctile)}%` }} />
        </span>
      )}
    </span>
  );
}

// ---- stat card ---------------------------------------------------------------

export function StatCard({
  label,
  value,
  sub,
  accent,
}: {
  label: string;
  value: React.ReactNode;
  sub?: string;
  accent?: string;
}) {
  return (
    <div className="rounded-lg border border-line bg-surface p-3.5">
      <div className="text-[10px] font-semibold uppercase tracking-wider text-ink-3">{label}</div>
      <div className={clsx("font-mono mt-1.5 text-[24px] font-bold leading-none tracking-tight", accent ?? "text-ink")}>
        {value}
      </div>
      {sub && <div className="mt-1.5 truncate font-mono text-[11px] text-ink-3">{sub}</div>}
    </div>
  );
}
