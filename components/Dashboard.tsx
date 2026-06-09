"use client";

import { useEffect, useState } from "react";
import clsx from "clsx";
import type { LiteModel } from "@/lib/types";
import { useFilters } from "@/lib/store";
import { SidebarContent } from "./Sidebar";
import { ModelTable } from "./ModelTable";
import { ActiveFilters } from "./ActiveFilters";

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
        placeholder="Search models — name, id, provider…"
        className="h-10 w-full rounded-md border border-line bg-surface pl-10 pr-12 text-sm text-ink outline-none transition-colors placeholder:text-ink-3 focus:border-ink"
      />
      <kbd className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 rounded border border-line bg-surface-2 px-1.5 py-0.5 font-mono text-[10px] text-ink-3">
        /
      </kbd>
    </div>
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
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <div className="mx-auto w-full max-w-[1200px] px-5 pb-12">
      <div className="mb-4 flex items-center gap-2">
        <button
          onClick={() => setMenuOpen(true)}
          className="flex size-10 items-center justify-center rounded-md border border-line bg-surface text-ink-2 lg:hidden"
          aria-label="Filters"
        >
          <svg viewBox="0 0 24 24" className="size-5" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M3 6h18M6 12h12M10 18h4" />
          </svg>
        </button>
        <SearchBar />
      </div>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[208px_1fr]">
        <aside className="hidden lg:block">
          <div className="sticky top-4">
            <SidebarContent providers={providers} stats={stats} />
          </div>
        </aside>

        <main className="min-w-0">
          <ActiveFilters />
          <ModelTable models={models} />
        </main>
      </div>

      {/* Mobile filter drawer */}
      <div className={clsx("fixed inset-0 z-50 lg:hidden", menuOpen ? "pointer-events-auto" : "pointer-events-none")}>
        <div
          onClick={() => setMenuOpen(false)}
          className={clsx("absolute inset-0 bg-black/30 transition-opacity", menuOpen ? "opacity-100" : "opacity-0")}
        />
        <div
          className={clsx(
            "absolute left-0 top-0 h-full w-[280px] overflow-y-auto border-r border-line bg-surface p-4 transition-transform",
            menuOpen ? "translate-x-0" : "-translate-x-full",
          )}
        >
          <div className="mb-4 flex items-center justify-between">
            <span className="font-mono text-sm font-semibold text-ink">Filters</span>
            <button onClick={() => setMenuOpen(false)} className="text-2xl text-ink-3">
              &times;
            </button>
          </div>
          <SidebarContent providers={providers} stats={stats} onNavigate={() => setMenuOpen(false)} />
        </div>
      </div>
    </div>
  );
}
