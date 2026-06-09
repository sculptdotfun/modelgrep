export function Footer() {
  return (
    <footer className="mt-12 border-t border-line">
      <div className="mx-auto flex w-full max-w-[1320px] flex-col gap-2 px-5 py-7 text-xs text-ink-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <span className="font-mono font-semibold text-ink-2">
            model<span className="text-brand">grep</span>
          </span>{" "}
          — find &amp; understand every LLM.
        </div>
        <div>Benchmarks: Artificial Analysis &amp; Design Arena · Live data via OpenRouter</div>
      </div>
    </footer>
  );
}
