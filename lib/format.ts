// Display + scoring helpers shared across the UI.

export function fmtPrice(v: number | null | undefined): string {
  if (v == null) return "—";
  if (v === 0) return "Free";
  if (v < 0.01) return `$${v.toFixed(4)}`;
  if (v < 1) return `$${v.toFixed(3)}`;
  return `$${v.toFixed(2)}`;
}

export function fmtContext(n: number | null | undefined): string {
  if (!n) return "—";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(n % 1_000_000 ? 1 : 0)}M`;
  if (n >= 1000) return `${Math.round(n / 1000)}K`;
  return `${n}`;
}

export function fmtThroughput(n: number | null | undefined): string {
  if (!n) return "—";
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${Math.round(n)}`;
}

export function fmtLatency(ms: number | null | undefined): string {
  if (ms == null) return "—";
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
}

export function fmtCutoff(s: string | null | undefined): string {
  if (!s) return "—";
  // "2025-01-31" -> "Jan 2025"
  const [y, m] = s.split("-");
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const mi = parseInt(m, 10) - 1;
  return m && months[mi] ? `${months[mi]} ${y}` : y;
}

export function pct(n: number | null | undefined, digits = 1): string {
  if (n == null) return "—";
  // AA eval values are 0..1 fractions; render as %
  return `${(n * 100).toFixed(digits)}%`;
}

// 0–100-ish quality tiers used for color coding.
export type Tier = "elite" | "high" | "mid" | "low" | "na";

export function intelTier(v: number | null | undefined): Tier {
  if (v == null) return "na";
  if (v >= 50) return "elite";
  if (v >= 35) return "high";
  if (v >= 20) return "mid";
  return "low";
}

export function eloTier(v: number | null | undefined): Tier {
  if (v == null) return "na";
  if (v >= 1200) return "elite";
  if (v >= 1050) return "high";
  if (v >= 900) return "mid";
  return "low";
}

export function speedTier(v: number | null | undefined): Tier {
  if (!v) return "na";
  if (v >= 200) return "elite";
  if (v >= 80) return "high";
  if (v >= 30) return "mid";
  return "low";
}

export function latencyTier(ms: number | null | undefined): Tier {
  if (ms == null) return "na";
  if (ms <= 400) return "elite";
  if (ms <= 1000) return "high";
  if (ms <= 2500) return "mid";
  return "low";
}

export function uptimeTier(v: number | null | undefined): Tier {
  if (v == null) return "na";
  if (v >= 99.5) return "elite";
  if (v >= 98) return "high";
  if (v >= 95) return "mid";
  return "low";
}

export const tierColor: Record<Tier, string> = {
  elite: "text-elite",
  high: "text-high",
  mid: "text-mid",
  low: "text-low",
  na: "text-ink-3",
};

// Pretty Design Arena category labels.
export const daCategoryLabel: Record<string, string> = {
  "3d": "3D",
  dataviz: "Data Viz",
  uicomponent: "UI Component",
  website: "Website",
  code: "Code",
  gamedev: "Game Dev",
  animation: "Animation",
};

export function modelOwner(id: string): string {
  return id.split("/")[0] ?? "";
}
