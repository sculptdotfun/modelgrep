"use client";

import { create } from "zustand";
import type { Model } from "./types";

export type SortKey =
  | "intelligence"
  | "coding"
  | "elo"
  | "throughput"
  | "latency"
  | "price_input"
  | "price_output"
  | "context_length";

export type CapKey = "tools" | "reasoning" | "vision" | "structured";

interface FilterState {
  query: string;
  sortKey: SortKey;
  sortDir: "asc" | "desc";
  caps: Record<CapKey, boolean>;
  freeOnly: boolean;
  benchOnly: boolean;
  maxPrice: number | null; // $/M input
  minContext: number | null; // tokens
  provider: string;
  preset: string;

  setQuery: (q: string) => void;
  setSort: (k: SortKey) => void;
  toggleCap: (c: CapKey) => void;
  setField: <K extends keyof FilterState>(k: K, v: FilterState[K]) => void;
  applyPreset: (p: string) => void;
  reset: () => void;
}

// Lower-is-better metrics sort ascending by default.
const ASC_DEFAULT: SortKey[] = ["latency", "price_input", "price_output"];

const initial = {
  query: "",
  sortKey: "intelligence" as SortKey,
  sortDir: "desc" as "asc" | "desc",
  caps: { tools: false, reasoning: false, vision: false, structured: false },
  freeOnly: false,
  benchOnly: false,
  maxPrice: null as number | null,
  minContext: null as number | null,
  provider: "",
  preset: "all",
};

export const useFilters = create<FilterState>((set) => ({
  ...initial,

  setQuery: (q) => set({ query: q, preset: "custom" }),

  setSort: (k) =>
    set((s) =>
      s.sortKey === k
        ? { sortDir: s.sortDir === "desc" ? "asc" : "desc" }
        : { sortKey: k, sortDir: ASC_DEFAULT.includes(k) ? "asc" : "desc" },
    ),

  toggleCap: (c) =>
    set((s) => ({ caps: { ...s.caps, [c]: !s.caps[c] }, preset: "custom" })),

  setField: (k, v) => set({ [k]: v, preset: "custom" } as Partial<FilterState>),

  applyPreset: (p) => {
    const base = { ...initial, preset: p };
    switch (p) {
      case "smartest":
        return set({ ...base, sortKey: "intelligence", sortDir: "desc", benchOnly: true });
      case "coding":
        return set({ ...base, sortKey: "coding", sortDir: "desc", benchOnly: true });
      case "design":
        return set({ ...base, sortKey: "elo", sortDir: "desc", benchOnly: true });
      case "fastest":
        return set({ ...base, sortKey: "throughput", sortDir: "desc" });
      case "lowlat":
        return set({ ...base, sortKey: "latency", sortDir: "asc" });
      case "cheapest":
        return set({ ...base, sortKey: "price_input", sortDir: "asc" });
      case "free":
        return set({ ...base, freeOnly: true, sortKey: "intelligence", sortDir: "desc" });
      case "vision":
        return set({ ...base, caps: { ...initial.caps, vision: true } });
      case "tools":
        return set({ ...base, caps: { ...initial.caps, tools: true } });
      case "longctx":
        return set({ ...base, minContext: 200_000, sortKey: "context_length", sortDir: "desc" });
      default:
        return set(base);
    }
  },

  reset: () => set(initial),
}));

// Pure selector: apply current filters to a catalog. Kept here so the table and
// any header counts share one definition.
export function selectModels(models: Model[], f: FilterState): Model[] {
  const terms = f.query.toLowerCase().split(/\s+/).filter(Boolean);

  const filtered = models.filter((m) => {
    if (terms.length) {
      const hay = `${m.id} ${m.name}`.toLowerCase();
      if (!terms.some((t) => hay.includes(t))) return false;
    }
    if (f.freeOnly && (m.price_input ?? 0) > 0) return false;
    if (f.benchOnly && !m.aa && !m.da) return false;
    if (f.maxPrice != null && (m.price_input ?? Infinity) > f.maxPrice) return false;
    if (f.minContext != null && (m.context_length ?? 0) < f.minContext) return false;
    if (f.provider && !m.providers.some((p) => p.toLowerCase().includes(f.provider.toLowerCase())))
      return false;
    for (const c of ["tools", "reasoning", "vision", "structured"] as CapKey[]) {
      if (f.caps[c] && !m.capabilities[c]) return false;
    }
    return true;
  });

  const val = (m: Model): number | null => {
    switch (f.sortKey) {
      case "intelligence":
        return m.aa?.intelligence ?? null;
      case "coding":
        return m.aa?.coding ?? null;
      case "elo":
        return m.da?.elo ?? null;
      case "throughput":
        return m.throughput || null;
      case "latency":
        return m.latency;
      case "price_input":
        return m.price_input;
      case "price_output":
        return m.price_output;
      case "context_length":
        return m.context_length || null;
    }
  };

  const dir = f.sortDir === "asc" ? 1 : -1;
  return filtered.sort((a, b) => {
    const av = val(a);
    const bv = val(b);
    // Nulls always sink to the bottom regardless of direction.
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    return (av - bv) * dir;
  });
}
