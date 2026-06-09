"use client";

import clsx from "clsx";
import { type CapKey, useFilters } from "@/lib/store";

const PRESETS: { id: string; label: string; hint?: string }[] = [
  { id: "all", label: "All models" },
  { id: "smartest", label: "Smartest", hint: "intel" },
  { id: "coding", label: "Best at code", hint: "code" },
  { id: "design", label: "Best at design", hint: "elo" },
  { id: "fastest", label: "Fastest", hint: "t/s" },
  { id: "lowlat", label: "Lowest latency", hint: "ms" },
  { id: "cheapest", label: "Cheapest", hint: "$" },
  { id: "free", label: "Free tier" },
  { id: "longctx", label: "Long context", hint: ">200K" },
];

const CAPS: { id: CapKey; label: string }[] = [
  { id: "reasoning", label: "Reasoning" },
  { id: "tools", label: "Tool calling" },
  { id: "structured", label: "Structured (JSON)" },
  { id: "vision", label: "Vision" },
];

export function SidebarContent({
  providers,
  stats,
  onNavigate,
}: {
  providers: string[];
  stats: { models: number; providers: number; benchmarked: number };
  onNavigate?: () => void;
}) {
  const f = useFilters();

  return (
    <div className="flex h-full flex-col gap-5">
      <div>
        <div className="mb-1.5 px-1 text-[10px] font-semibold uppercase tracking-wider text-ink-3">Presets</div>
        <div className="flex flex-col gap-0.5">
          {PRESETS.map((p) => (
            <button
              key={p.id}
              onClick={() => {
                f.applyPreset(p.id);
                onNavigate?.();
              }}
              className={clsx(
                "flex items-center justify-between rounded-md px-2.5 py-1.5 text-[13px] transition-colors",
                f.preset === p.id
                  ? "bg-surface-2 font-semibold text-ink"
                  : "text-ink-2 hover:bg-surface-2/60 hover:text-ink",
              )}
            >
              {p.label}
              {p.hint && <span className="font-mono text-[10px] text-ink-3">{p.hint}</span>}
            </button>
          ))}
        </div>
      </div>

      <div>
        <div className="mb-1.5 px-1 text-[10px] font-semibold uppercase tracking-wider text-ink-3">Capabilities</div>
        <div className="flex flex-col gap-0.5">
          {CAPS.map((c) => (
            <label key={c.id} className="flex cursor-pointer items-center gap-2.5 rounded-md px-2.5 py-1.5 text-[13px] text-ink-2 hover:bg-surface-2">
              <input type="checkbox" checked={f.caps[c.id]} onChange={() => f.toggleCap(c.id)} className="size-3.5 accent-[#101014]" />
              {c.label}
            </label>
          ))}
          <label className="flex cursor-pointer items-center gap-2.5 rounded-md px-2.5 py-1.5 text-[13px] text-ink-2 hover:bg-surface-2">
            <input type="checkbox" checked={f.benchOnly} onChange={() => f.setField("benchOnly", !f.benchOnly)} className="size-3.5 accent-[#101014]" />
            Has benchmarks
          </label>
          <label className="flex cursor-pointer items-center gap-2.5 rounded-md px-2.5 py-1.5 text-[13px] text-ink-2 hover:bg-surface-2">
            <input type="checkbox" checked={f.freeOnly} onChange={() => f.setField("freeOnly", !f.freeOnly)} className="size-3.5 accent-[#101014]" />
            Free only
          </label>
        </div>
      </div>

      <div>
        <div className="mb-1.5 px-1 text-[10px] font-semibold uppercase tracking-wider text-ink-3">Refine</div>
        <div className="flex flex-col gap-2 px-1">
          <label className="flex items-center justify-between gap-2 text-[13px] text-ink-2">
            <span>Max input $/M</span>
            <input
              type="number"
              min={0}
              step={0.5}
              value={f.maxPrice ?? ""}
              onChange={(e) => f.setField("maxPrice", e.target.value === "" ? null : Number(e.target.value))}
              placeholder="any"
              className="w-16 rounded-md border border-line bg-surface px-2 py-1 text-right font-mono text-[12px] text-ink outline-none focus:border-ink"
            />
          </label>
          <label className="flex items-center justify-between gap-2 text-[13px] text-ink-2">
            <span>Min context</span>
            <span className="flex items-center gap-1">
              <input
                type="number"
                min={0}
                step={32}
                value={f.minContext != null ? Math.round(f.minContext / 1000) : ""}
                onChange={(e) => f.setField("minContext", e.target.value === "" ? null : Number(e.target.value) * 1000)}
                placeholder="any"
                className="w-14 rounded-md border border-line bg-surface px-2 py-1 text-right font-mono text-[12px] text-ink outline-none focus:border-ink"
              />
              <span className="font-mono text-[11px] text-ink-3">K</span>
            </span>
          </label>
        </div>
      </div>

      {providers.length > 0 && (
        <div>
          <div className="mb-1.5 px-1 text-[10px] font-semibold uppercase tracking-wider text-ink-3">Provider</div>
          <select
            value={f.provider}
            onChange={(e) => f.setField("provider", e.target.value)}
            className="w-full rounded-md border border-line bg-surface px-2.5 py-1.5 text-[13px] text-ink-2 outline-none focus:border-ink"
          >
            <option value="">All providers</option>
            {providers.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </div>
      )}

      <div className="card-shadow mt-auto rounded-lg border border-line bg-surface p-3">
        <Stat label="Models" value={stats.models} />
        <Stat label="Providers" value={stats.providers} />
        <Stat label="Benchmarked" value={stats.benchmarked} />
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex items-center justify-between py-0.5 text-xs">
      <span className="text-ink-3">{label}</span>
      <span className="font-mono font-medium text-ink">{value}</span>
    </div>
  );
}
