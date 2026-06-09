"use client";

import { type CapKey, useFilters } from "@/lib/store";
import { fmtContext, fmtPrice } from "@/lib/format";

const CAP_LABEL: Record<CapKey, string> = {
  reasoning: "Reasoning",
  tools: "Tool calling",
  vision: "Vision",
  structured: "Structured (JSON)",
};

function Chip({ label, onRemove }: { label: string; onRemove: () => void }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-md border border-line bg-surface px-2.5 py-1 text-[12px] text-ink-2">
      {label}
      <button onClick={onRemove} className="-mr-0.5 ml-0.5 text-ink-3 hover:text-low" aria-label={`Remove ${label}`}>
        ✕
      </button>
    </span>
  );
}

export function ActiveFilters() {
  const f = useFilters();

  const chips: { key: string; label: string; remove: () => void }[] = [];
  if (f.query) chips.push({ key: "q", label: `“${f.query}”`, remove: () => f.setQuery("") });
  for (const c of ["reasoning", "tools", "vision", "structured"] as CapKey[]) {
    if (f.caps[c]) chips.push({ key: c, label: CAP_LABEL[c], remove: () => f.toggleCap(c) });
  }
  if (f.freeOnly) chips.push({ key: "free", label: "Free only", remove: () => f.setField("freeOnly", false) });
  if (f.benchOnly) chips.push({ key: "bench", label: "Benchmarked", remove: () => f.setField("benchOnly", false) });
  if (f.maxPrice != null)
    chips.push({ key: "price", label: `≤ ${fmtPrice(f.maxPrice)}/M`, remove: () => f.setField("maxPrice", null) });
  if (f.minContext != null)
    chips.push({ key: "ctx", label: `≥ ${fmtContext(f.minContext)} ctx`, remove: () => f.setField("minContext", null) });

  if (chips.length === 0) return null;

  return (
    <div className="mb-3 flex flex-wrap items-center gap-2">
      <span className="text-[11px] font-medium uppercase tracking-wider text-ink-3">Filters</span>
      {chips.map((c) => (
        <Chip key={c.key} label={c.label} onRemove={c.remove} />
      ))}
      {chips.length > 1 && (
        <button onClick={() => f.reset()} className="text-[12px] font-medium text-brand-ink hover:underline">
          Clear all
        </button>
      )}
    </div>
  );
}
