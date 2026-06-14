"use client";

import { useEffect, useState } from "react";
import clsx from "clsx";
import type { LiteModel } from "@/lib/types";
import { type CapKey, useFilters } from "@/lib/store";
import { ModelTable } from "./ModelTable";
import { ActiveFilters } from "./ActiveFilters";

// Presets mirror the /best collections — one-tap views over the catalog.
const PRESETS: { id: string; label: string }[] = [
  { id: "all", label: "All" },
  { id: "smartest", label: "Smartest" },
  { id: "coding", label: "Coding" },
  { id: "design", label: "Design" },
  { id: "fastest", label: "Fastest" },
  { id: "lowlat", label: "Low latency" },
  { id: "cheapest", label: "Cheapest" },
  { id: "free", label: "Free" },
  { id: "longctx", label: "Long context" },
];

const CAPS: { id: CapKey; label: string }[] = [
  { id: "reasoning", label: "Reasoning" },
  { id: "tools", label: "Tools" },
  { id: "vision", label: "Vision" },
  { id: "structured", label: "JSON" },
];

function SearchBar() {
  const { query, setQuery } = useFilters();
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "/" && !(e.target as HTMLElement).matches("input,textarea,select")) {
        e.preventDefault();
        document.getElementById("mg-search")?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  return (
    <div className="relative flex-1">
      <svg className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-ink-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="11" cy="11" r="8" />
        <path d="m21 21-4.35-4.35" />
      </svg>
      <input
        id="mg-search"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search 300+ models — name, id, provider…"
        className="h-10 w-full rounded-lg border border-line bg-surface pl-10 pr-12 text-sm text-ink outline-none transition-colors placeholder:text-ink-3 focus:border-ink"
      />
      <kbd className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 rounded border border-line bg-surface-2 px-1.5 py-0.5 font-mono text-[10px] text-ink-3">
        /
      </kbd>
    </div>
  );
}

function Pill({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={clsx(
        "shrink-0 whitespace-nowrap rounded-full border px-3 py-1.5 text-[12.5px] font-medium transition-colors",
        active
          ? "border-ink bg-ink text-white"
          : "border-line bg-surface text-ink-2 hover:border-line-strong hover:text-ink",
      )}
    >
      {children}
    </button>
  );
}

function NumberField({
  label,
  suffix,
  value,
  onChange,
  step,
  placeholder = "any",
}: {
  label: string;
  suffix?: string;
  value: number | "";
  onChange: (v: number | null) => void;
  step?: number;
  placeholder?: string;
}) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-[11px] font-semibold uppercase tracking-wider text-ink-3">{label}</span>
      <span className="flex items-center gap-1.5">
        <input
          type="number"
          min={0}
          step={step}
          value={value}
          onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))}
          placeholder={placeholder}
          className="h-9 w-full rounded-md border border-line bg-surface px-2.5 text-right font-mono text-[13px] text-ink outline-none focus:border-ink"
        />
        {suffix && <span className="shrink-0 font-mono text-[12px] text-ink-3">{suffix}</span>}
      </span>
    </label>
  );
}

function Toggle({ checked, onChange, label }: { checked: boolean; onChange: () => void; label: string }) {
  return (
    <label className="flex cursor-pointer items-center gap-2.5 text-[13px] text-ink-2">
      <input type="checkbox" checked={checked} onChange={onChange} className="size-4 accent-[#6d4aff]" />
      {label}
    </label>
  );
}

export function Dashboard({
  models,
  providers,
  stats,
}: {
  models: LiteModel[];
  providers: string[];
  stats: { models: number; providers: number; benchmarked: number };
}) {
  const f = useFilters();
  const [showFilters, setShowFilters] = useState(false);
  const advancedActive = f.maxPrice != null || f.minContext != null || Boolean(f.provider);

  return (
    <div className="mx-auto w-full max-w-[1200px] px-5 pb-12">
      {/* Command bar — search, presets, capability toggles, advanced filters */}
      <div className="card-shadow rounded-xl border border-line bg-surface p-3">
        <div className="flex items-center gap-2">
          <SearchBar />
          <button
            onClick={() => setShowFilters((v) => !v)}
            className={clsx(
              "flex h-10 shrink-0 items-center gap-1.5 rounded-lg border px-3 text-[13px] font-medium transition-colors",
              showFilters || advancedActive ? "border-ink text-ink" : "border-line text-ink-2 hover:text-ink",
            )}
          >
            <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 6h18M7 12h10M11 18h2" />
            </svg>
            <span className="hidden sm:inline">Filters</span>
            {advancedActive && <span className="size-1.5 rounded-full bg-brand" />}
          </button>
        </div>

        {/* Preset + capability pills */}
        <div className="mt-3 flex items-center gap-1.5 overflow-x-auto pb-0.5 [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
          {PRESETS.map((p) => (
            <Pill key={p.id} active={f.preset === p.id} onClick={() => f.applyPreset(p.id)}>
              {p.label}
            </Pill>
          ))}
          <span className="mx-1 h-5 w-px shrink-0 bg-line" />
          {CAPS.map((c) => (
            <Pill key={c.id} active={f.caps[c.id]} onClick={() => f.toggleCap(c.id)}>
              {c.label}
            </Pill>
          ))}
        </div>

        {/* Advanced filters */}
        {showFilters && (
          <div className="mt-3 border-t border-line pt-3.5">
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <NumberField
                label="Max input $/M"
                value={f.maxPrice ?? ""}
                step={0.5}
                onChange={(v) => f.setField("maxPrice", v)}
              />
              <NumberField
                label="Min context"
                suffix="K"
                step={32}
                value={f.minContext != null ? Math.round(f.minContext / 1000) : ""}
                onChange={(v) => f.setField("minContext", v == null ? null : v * 1000)}
              />
              {providers.length > 0 && (
                <label className="flex flex-col gap-1.5">
                  <span className="text-[11px] font-semibold uppercase tracking-wider text-ink-3">Provider</span>
                  <select
                    value={f.provider}
                    onChange={(e) => f.setField("provider", e.target.value)}
                    className="h-9 w-full rounded-md border border-line bg-surface px-2 text-[13px] text-ink-2 outline-none focus:border-ink"
                  >
                    <option value="">All providers</option>
                    {providers.map((p) => (
                      <option key={p} value={p}>
                        {p}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              <div className="flex flex-col justify-end gap-2 pb-1">
                <Toggle checked={f.freeOnly} onChange={() => f.setField("freeOnly", !f.freeOnly)} label="Free only" />
                <Toggle checked={f.benchOnly} onChange={() => f.setField("benchOnly", !f.benchOnly)} label="Has benchmarks" />
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="mt-4">
        <ActiveFilters />
        <ModelTable models={models} benchmarked={stats.benchmarked} />
      </div>
    </div>
  );
}
