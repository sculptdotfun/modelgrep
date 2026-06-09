"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import clsx from "clsx";
import type { LiteModel } from "@/lib/types";
import { ownerColor } from "@/lib/owners";
import { fmtPrice, fmtThroughput, modelOwner } from "@/lib/format";

type MetricKey = "intelligence" | "coding" | "elo" | "throughput";

const METRICS: { key: MetricKey; label: string }[] = [
  { key: "intelligence", label: "Intelligence" },
  { key: "coding", label: "Coding" },
  { key: "elo", label: "Design" },
  { key: "throughput", label: "Speed" },
];

function metricValue(m: LiteModel, k: MetricKey): number | null {
  switch (k) {
    case "intelligence":
      return m.aa?.intelligence ?? null;
    case "coding":
      return m.aa?.coding ?? null;
    case "elo":
      return m.da?.elo ?? null;
    case "throughput":
      return m.throughput || null;
  }
}

function display(m: LiteModel, k: MetricKey): string {
  const v = metricValue(m, k);
  if (v == null) return "—";
  if (k === "throughput") return `${fmtThroughput(v)} t/s`;
  if (k === "elo") return String(v);
  return v.toFixed(1);
}

export function TopChart({ models }: { models: LiteModel[] }) {
  const router = useRouter();
  const [metric, setMetric] = useState<MetricKey>("intelligence");

  const { rows, max } = useMemo(() => {
    const ranked = models
      .map((m) => ({ m, v: metricValue(m, metric) }))
      .filter((r): r is { m: LiteModel; v: number } => r.v != null)
      .sort((a, b) => b.v - a.v)
      .slice(0, 10);
    const max = ranked[0]?.v ?? 1;
    // Design Elo bars look better scaled from a floor rather than 0.
    const floor = metric === "elo" ? 800 : 0;
    return { rows: ranked, max, floor } as { rows: typeof ranked; max: number; floor: number };
  }, [models, metric]);

  const floor = metric === "elo" ? 800 : 0;

  return (
    <div className="rounded-lg border border-line bg-surface p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="font-mono text-[11px] font-semibold uppercase tracking-widest text-ink-2">Top models</h2>
        <div className="flex overflow-hidden rounded-md border border-line">
          {METRICS.map((mt) => (
            <button
              key={mt.key}
              onClick={() => setMetric(mt.key)}
              className={clsx(
                "px-2.5 py-1 text-[11px] font-medium transition-colors",
                metric === mt.key ? "bg-ink text-white" : "text-ink-3 hover:bg-surface-2 hover:text-ink",
              )}
            >
              {mt.label}
            </button>
          ))}
        </div>
      </div>

      <div className="flex flex-col">
        {rows.map(({ m, v }, i) => {
          const oc = ownerColor(modelOwner(m.id));
          const w = max > floor ? ((v - floor) / (max - floor)) * 100 : 100;
          const name = m.id.split("/").slice(1).join("/");
          return (
            <button
              key={m.id}
              onClick={() => router.push(`/models/${m.id}`)}
              className="group flex items-center gap-2.5 rounded-md px-1.5 py-[7px] text-left transition-colors hover:bg-surface-2/70"
            >
              <span className="w-4 shrink-0 text-right font-mono text-[11px] tabular-nums text-ink-3">{i + 1}</span>
              <span className="size-2 shrink-0 rounded-[2px]" style={{ background: oc }} />
              <span className="w-28 shrink-0 truncate font-mono text-[12px] text-ink group-hover:text-brand-ink" title={m.id}>
                {name}
              </span>
              <span className="relative h-2 flex-1 overflow-hidden rounded-[1px] bg-surface-2">
                <span
                  key={metric}
                  className="animate-grow absolute inset-y-0 left-0 rounded-[1px]"
                  style={{ width: `${Math.max(3, w)}%`, background: oc, opacity: 0.85 }}
                />
              </span>
              <span className="w-16 shrink-0 whitespace-nowrap text-right font-mono text-[12px] font-semibold tabular-nums text-ink">
                {display(m, metric)}
              </span>
            </button>
          );
        })}
      </div>

      <div className="mt-2 px-1.5 text-[11px] text-ink-3">
        {metric === "throughput"
          ? "Fastest by tokens/sec"
          : metric === "elo"
            ? "Best at frontend/design generation (Design Arena Elo)"
            : metric === "coding"
              ? "Artificial Analysis Coding Index"
              : "Artificial Analysis Intelligence Index"}
      </div>
    </div>
  );
}
