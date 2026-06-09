"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";
import clsx from "clsx";
import type { LiteModel } from "@/lib/types";
import { type SortKey, selectModels, useFilters } from "@/lib/store";
import { ownerColor } from "@/lib/owners";
import {
  daCategoryLabel,
  fmtContext,
  fmtLatency,
  fmtPrice,
  fmtThroughput,
  intelTier,
  latencyTier,
  modelOwner,
  type Tier,
} from "@/lib/format";
import { CapIcons } from "./ui";

interface Col {
  key: SortKey;
  label: string;
  hideAt?: string;
}

const COLS: Col[] = [
  { key: "coding", label: "Code", hideAt: "hidden lg:table-cell" },
  { key: "elo", label: "Design", hideAt: "hidden xl:table-cell" },
  { key: "throughput", label: "Speed" },
  { key: "latency", label: "Latency", hideAt: "hidden md:table-cell" },
  { key: "price_input", label: "In $/M" },
  { key: "price_output", label: "Out $/M", hideAt: "hidden sm:table-cell" },
  { key: "context_length", label: "Context" },
];

const BAR_FILL: Record<Tier, string> = {
  elite: "bg-elite",
  high: "bg-high",
  mid: "bg-mid",
  low: "bg-low",
  na: "bg-ink-3",
};

// Plain ranks; the top three carry weight through type, not decoration.
function MedalRank({ rank }: { rank: number }) {
  return (
    <span className={clsx("font-mono text-xs tabular-nums", rank <= 3 ? "font-bold text-ink" : "text-ink-3")}>
      {rank}
    </span>
  );
}

function SortHeader({ col, alwaysRight = true }: { col: Col; alwaysRight?: boolean }) {
  const { sortKey, sortDir, setSort } = useFilters();
  const active = sortKey === col.key;
  return (
    <th className={clsx("select-none px-3 py-2.5 text-[10px] font-semibold uppercase tracking-wider", alwaysRight && "text-right", col.hideAt)}>
      <button
        onClick={() => setSort(col.key)}
        className={clsx("inline-flex items-center gap-1 transition-colors", active ? "text-ink" : "text-ink-3 hover:text-ink-2")}
      >
        {col.label}
        <span className={clsx("text-[7px]", active ? "opacity-100" : "opacity-0")}>{sortDir === "desc" ? "▼" : "▲"}</span>
      </button>
    </th>
  );
}

// number cell helpers — neutral ink hierarchy, color reserved for real signal.
const NUM = "font-mono text-[13px] tabular-nums";

function Row({ m, rank }: { m: LiteModel; rank: number }) {
  const router = useRouter();
  const href = `/models/${m.id}`;
  const owner = modelOwner(m.id);
  const rest = m.id.slice(owner.length); // "/gpt-4o"
  const oc = ownerColor(owner);

  const intel = m.aa?.intelligence ?? null;
  const coding = m.aa?.coding ?? null;
  const elo = m.da?.elo ?? null;
  const it = intelTier(intel);
  const slow = latencyTier(m.latency) === "low";
  const free = (m.price_input ?? -1) === 0;

  return (
    <tr onClick={() => router.push(href)} className="group cursor-pointer border-t border-line transition-colors hover:bg-surface-2/60">
      <td className="w-10 py-3 pl-4 pr-1 text-center align-middle">
        <MedalRank rank={rank} />
      </td>

      <td className="py-3 pr-3 align-middle">
        <div className="flex items-center gap-2.5">
          <span className="size-2.5 shrink-0 rounded-[2px]" style={{ background: oc }} />
          <Link href={href} onClick={(e) => e.stopPropagation()} className="font-mono text-[13.5px] leading-none">
            <span className="text-ink-3">{owner}</span>
            <span className="font-semibold text-ink group-hover:text-brand-ink">{rest}</span>
          </Link>
          <CapIcons caps={m.capabilities} />
        </div>
      </td>

      {/* Intelligence — the anchor metric, the only ranked color (heatmap down the list) */}
      <td className="py-3 pr-4 align-middle">
        <div className="ml-auto flex w-[128px] items-center gap-2.5">
          <div className="h-[5px] flex-1 overflow-hidden rounded-[1px] bg-surface-2">
            {intel != null && (
              <div className={clsx("h-full rounded-[1px]", BAR_FILL[it])} style={{ width: `${Math.min(100, (intel / 65) * 100)}%` }} />
            )}
          </div>
          <span className={clsx("w-9 text-right font-bold", NUM, intel != null ? "text-ink" : "text-ink-3")}>
            {intel != null ? intel.toFixed(1) : "—"}
          </span>
        </div>
      </td>

      <td className={clsx("hidden px-3 py-3 text-right align-middle lg:table-cell", NUM, coding != null ? "text-ink" : "text-ink-3")}>
        {coding != null ? coding.toFixed(1) : "—"}
      </td>
      <td className="hidden px-3 py-3 text-right align-middle xl:table-cell">
        {elo != null ? (
          <div className="flex flex-col items-end leading-none">
            <span className={clsx(NUM, "font-medium text-ink")}>{elo}</span>
            {m.da?.category && <span className="mt-1 text-[9px] text-ink-3">{daCategoryLabel[m.da.category] ?? m.da.category}</span>}
          </div>
        ) : (
          <span className={clsx(NUM, "text-ink-3")}>—</span>
        )}
      </td>

      <td className={clsx("px-3 py-3 text-right align-middle", NUM, m.throughput ? "text-ink" : "text-ink-3")}>
        {m.throughput ? fmtThroughput(m.throughput) : "—"}
      </td>
      <td className={clsx("hidden px-3 py-3 text-right align-middle md:table-cell", NUM, m.latency == null ? "text-ink-3" : slow ? "text-low" : "text-ink-2")}>
        {fmtLatency(m.latency)}
      </td>
      <td className={clsx("px-3 py-3 text-right align-middle", NUM, free ? "font-semibold text-elite" : "text-ink")}>
        {fmtPrice(m.price_input)}
      </td>
      <td className={clsx("hidden px-3 py-3 text-right align-middle sm:table-cell", NUM, "text-ink-2")}>{fmtPrice(m.price_output)}</td>
      <td className={clsx("px-3 py-3 pr-4 text-right align-middle", NUM, "text-ink-2")}>{fmtContext(m.context_length)}</td>
    </tr>
  );
}

// Rows rendered before "Show all" — keeps SSR HTML and hydration light.
const INITIAL_ROWS = 60;

export function ModelTable({ models }: { models: LiteModel[] }) {
  const filters = useFilters();
  const [showAll, setShowAll] = useState(false);
  const rows = useMemo(() => selectModels(models, filters), [models, filters]);
  const visible = showAll ? rows : rows.slice(0, INITIAL_ROWS);

  return (
    <div className="rounded-lg border border-line bg-surface">
      <div className="flex items-center justify-between rounded-t-lg border-b border-line px-4 py-2.5 text-xs text-ink-2">
        <span>
          <strong className="text-ink">{rows.length}</strong> models
        </span>
        <span className="hidden font-mono text-[11px] text-ink-3 sm:inline">
          ranked by {filters.sortKey.replace("_", " ")} · click a row →
        </span>
      </div>
      <div>
        <table className="w-full border-collapse">
          <thead className="sticky top-0 z-20 bg-surface/95 backdrop-blur">
            <tr className="border-b border-line">
              <th className="w-10 py-2.5 pl-4 pr-1 text-center text-[10px] font-semibold uppercase tracking-wider text-ink-3">#</th>
              <th className="py-2.5 pr-3 text-left text-[10px] font-semibold uppercase tracking-wider text-ink-3">Model</th>
              <th className="py-2.5 pr-4 text-right">
                <button
                  onClick={() => filters.setSort("intelligence")}
                  className={clsx(
                    "inline-flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wider transition-colors",
                    filters.sortKey === "intelligence" ? "text-ink" : "text-ink-3 hover:text-ink-2",
                  )}
                >
                  Intelligence
                  <span className={clsx("text-[7px]", filters.sortKey === "intelligence" ? "opacity-100" : "opacity-0")}>
                    {filters.sortDir === "desc" ? "▼" : "▲"}
                  </span>
                </button>
              </th>
              {COLS.map((c) => (
                <SortHeader key={c.key} col={c} />
              ))}
            </tr>
          </thead>
          <tbody>
            {visible.map((m, i) => (
              <Row key={m.id} m={m} rank={i + 1} />
            ))}
          </tbody>
        </table>
        {rows.length === 0 && <div className="py-16 text-center text-sm text-ink-3">No models match your filters.</div>}
        {!showAll && rows.length > INITIAL_ROWS && (
          <button
            onClick={() => setShowAll(true)}
            className="block w-full border-t border-line py-3 text-center text-[13px] font-medium text-brand-ink transition-colors hover:bg-surface-2/60"
          >
            Show all {rows.length} models
          </button>
        )}
      </div>
    </div>
  );
}
