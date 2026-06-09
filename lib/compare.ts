// Metric definitions for side-by-side model comparison.
import type { Model } from "./types";

export interface CompareMetric {
  key: string;
  label: string;
  // value extractor; null when unavailable
  get: (m: Model) => number | null;
  // formatted display
  fmt: (v: number | null) => string;
  // true when higher is better
  higherBetter: boolean;
}

import { fmtContext, fmtLatency, fmtPrice, fmtThroughput, pct } from "./format";

export const COMPARE_METRICS: CompareMetric[] = [
  { key: "intelligence", label: "Intelligence Index", get: (m) => m.aa?.intelligence ?? null, fmt: (v) => (v == null ? "—" : v.toFixed(1)), higherBetter: true },
  { key: "coding", label: "Coding Index", get: (m) => m.aa?.coding ?? null, fmt: (v) => (v == null ? "—" : v.toFixed(1)), higherBetter: true },
  { key: "gpqa", label: "GPQA Diamond", get: (m) => m.aa?.gpqa ?? null, fmt: (v) => (v == null ? "—" : pct(v, 0)), higherBetter: true },
  { key: "elo", label: "Design Arena Elo", get: (m) => m.da?.elo ?? null, fmt: (v) => (v == null ? "—" : String(v)), higherBetter: true },
  { key: "throughput", label: "Speed (tokens/sec)", get: (m) => m.throughput || null, fmt: (v) => (v == null ? "—" : fmtThroughput(v)), higherBetter: true },
  { key: "latency", label: "Latency", get: (m) => m.latency, fmt: (v) => fmtLatency(v), higherBetter: false },
  { key: "price_input", label: "Input price /M", get: (m) => m.price_input, fmt: (v) => fmtPrice(v), higherBetter: false },
  { key: "price_output", label: "Output price /M", get: (m) => m.price_output, fmt: (v) => fmtPrice(v), higherBetter: false },
  { key: "context", label: "Context window", get: (m) => m.context_length || null, fmt: (v) => fmtContext(v), higherBetter: true },
];

// -1 = A wins, 1 = B wins, 0 = tie/unknown
export function winner(metric: CompareMetric, a: Model, b: Model): -1 | 0 | 1 {
  const av = metric.get(a);
  const bv = metric.get(b);
  if (av == null && bv == null) return 0;
  if (av == null) return 1;
  if (bv == null) return -1;
  if (av === bv) return 0;
  const aBetter = metric.higherBetter ? av > bv : av < bv;
  return aBetter ? -1 : 1;
}
