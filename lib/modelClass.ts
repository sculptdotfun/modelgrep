// Model "class" axis — the size/tier dimension the catalog has no field for.
//
// Closed labs (Anthropic, OpenAI, Google) don't publish parameter counts, so
// "small" can't be read off a number. It has to be inferred from the signals we
// do have: the efficiency tier is the cheap + fast quadrant, reinforced by the
// naming labs use for it (haiku, mini, flash, nano, lite). This is what lets the
// site answer "small fast anthropic model" → Claude Haiku instead of shrugging.

import type { Model } from "./types";

export type ModelClass = "frontier" | "balanced" | "small";

// Tokens labs use to name their small/efficient *tier*. Deliberately excludes
// serving-speed words like "fast"/"turbo"/"instant" — "Claude Opus (Fast)" is a
// fast-served flagship, not a small model. Word-boundaried so "flash" matches
// "Gemini Flash" but not a random substring.
const SMALL_TOKENS =
  /(?:^|[\s\-./:])(haiku|mini|flash|nano|lite|small|tiny|micro|air|scout)(?:$|[\s\-./:0-9])/i;
// Explicit small parameter counts in the name: 1B–14B.
const SMALL_PARAMS = /\b(?:[1-9]|1[0-4])b\b/i;
// Tokens that imply the flagship/large tier.
const FRONTIER_TOKENS =
  /(?:^|[\s\-./:])(opus|ultra|max|large|huge|70b|72b|123b|235b|405b|480b|671b)(?:$|[\s\-./:])/i;

function nameOf(m: Model): string {
  return `${m.id} ${m.name}`;
}

function quantile(sorted: number[], p: number): number {
  if (!sorted.length) return Infinity;
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))];
}

interface Thresholds {
  cheapInput: number; // input price at/below which a model counts as "cheap"
  fastTput: number; // throughput at/above which a model counts as "fast"
}

function classify(m: Model, t: Thresholds): ModelClass {
  const name = nameOf(m);
  const intel = m.aa?.intelligence ?? null;

  const namedSmall = SMALL_TOKENS.test(name) || SMALL_PARAMS.test(name);
  const namedFrontier = FRONTIER_TOKENS.test(name);

  // A genuinely frontier-intelligent model isn't "small" even if it's cheap/fast.
  if (intel != null && intel >= 50) return "frontier";
  // A flagship-named model (Opus/Ultra/Max/Large/70B+) is never "small", even
  // when a serving variant lacks a benchmark score. This is what keeps
  // "Claude Opus (Fast)" out of the small-and-fast list.
  if (namedFrontier && !namedSmall) return "frontier";

  // Past the frontier guards: any model named for the small tier is small.
  if (namedSmall) return "small";

  // Data-driven small: cheaper than the median paid model AND faster than the
  // median model AND not in frontier intelligence territory.
  const cheap = (m.price_input ?? Infinity) <= t.cheapInput;
  const fast = m.throughput > 0 && m.throughput >= t.fastTput;
  if (cheap && fast && (intel == null || intel < 40)) return "small";

  return "balanced";
}

// Build a class lookup for the whole catalog in one pass, using field-relative
// thresholds (so "cheap"/"fast" track the actual distribution, not magic numbers).
export function classIndex(models: Model[]): Map<string, ModelClass> {
  const prices = models
    .map((m) => m.price_input)
    .filter((v): v is number => v != null && v > 0)
    .sort((a, b) => a - b);
  const tputs = models
    .map((m) => m.throughput)
    .filter((v) => v > 0)
    .sort((a, b) => a - b);

  const t: Thresholds = {
    cheapInput: quantile(prices, 0.5),
    fastTput: quantile(tputs, 0.5),
  };

  const idx = new Map<string, ModelClass>();
  for (const m of models) idx.set(m.id, classify(m, t));
  return idx;
}

export const CLASS_LABEL: Record<ModelClass, string> = {
  frontier: "Frontier",
  balanced: "Balanced",
  small: "Small & fast",
};
