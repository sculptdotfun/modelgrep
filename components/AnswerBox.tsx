import { Sparkles } from "lucide-react";

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
    <div className="card-shadow hero-glow relative mt-5 overflow-hidden rounded-2xl border border-line bg-surface p-5 sm:p-6">
      <div className="mb-3 flex items-center gap-2">
        <span className="flex size-5 items-center justify-center rounded-md bg-brand/10 text-brand">
          <Sparkles className="size-3" strokeWidth={2.5} />
        </span>
        <span className="font-mono text-[10px] font-semibold uppercase tracking-widest text-ink-3">
          Quick answer · Updated {updated}
        </span>
      </div>
      <p className="max-w-3xl text-[16px] leading-relaxed text-ink sm:text-[17px]">{answer}</p>
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
