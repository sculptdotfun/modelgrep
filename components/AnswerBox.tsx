// AEO "quick answer" block. The single most-liftable asset on a ranking page:
// a BLUF declarative sentence plus a visible freshness stamp and the winner's
// key stats. AI answer engines extract exactly this kind of stat-dense,
// up-front, named-entity sentence.

export function AnswerBox({
  answer,
  updated,
  stats,
}: {
  answer: string;
  updated: string;
  stats?: { label: string; value: string }[];
}) {
  return (
    <div className="card-shadow mt-5 rounded-xl border border-line bg-surface p-5">
      <div className="mb-2.5 flex items-center gap-2">
        <span className="size-1.5 rounded-full bg-elite" />
        <span className="font-mono text-[10px] font-semibold uppercase tracking-widest text-ink-3">
          Quick answer · Updated {updated}
        </span>
      </div>
      <p className="max-w-3xl text-[16px] leading-relaxed text-ink">{answer}</p>
      {stats && stats.length > 0 && (
        <div className="mt-4 flex flex-wrap gap-x-6 gap-y-2 border-t border-line pt-3.5">
          {stats.map((s) => (
            <div key={s.label} className="flex items-baseline gap-1.5">
              <span className="font-mono text-[15px] font-bold tabular-nums text-ink">{s.value}</span>
              <span className="text-[11px] uppercase tracking-wide text-ink-3">{s.label}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
