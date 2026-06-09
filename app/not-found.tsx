import Link from "next/link";
import { Footer } from "@/components/Footer";
import { SiteHeader } from "@/components/SiteHeader";

export default function NotFound() {
  return (
    <div className="min-h-screen">
      <div className="mx-auto w-full max-w-[1200px] px-5 py-7">
        <SiteHeader />
        <div className="flex flex-col items-center py-28 text-center">
          <div className="font-mono text-[13px] font-semibold uppercase tracking-widest text-ink-3">404</div>
          <h1 className="font-display mt-3 text-[32px] font-bold text-ink">No matches found.</h1>
          <p className="mt-3 max-w-md text-[15px] leading-relaxed text-ink-2">
            That page doesn&apos;t exist — the model may have been renamed or retired.
          </p>
          <div className="mt-7 flex gap-2">
            <Link href="/" className="rounded-md bg-ink px-4 py-2 text-sm font-medium text-white transition-opacity hover:opacity-90">
              Browse the leaderboard
            </Link>
            <Link
              href="/new"
              className="rounded-md border border-line bg-surface px-4 py-2 text-sm font-medium text-ink-2 transition-colors hover:border-line-strong hover:text-ink"
            >
              New models
            </Link>
          </div>
        </div>
      </div>
      <Footer />
    </div>
  );
}
