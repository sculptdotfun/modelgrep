import type { Metadata } from "next";
import Link from "next/link";
import { GLOSSARY } from "@/lib/glossary";
import { Footer } from "@/components/Footer";
import { SiteHeader } from "@/components/SiteHeader";

export const metadata: Metadata = {
  title: "LLM Glossary — Context Windows, Benchmarks, Tokens & More",
  description:
    "Plain-English definitions of the terms that matter when comparing AI models: context windows, tokens per second, GPQA, Elo ratings, prompt caching, quantization and more.",
  alternates: { canonical: "/glossary" },
  openGraph: {
    title: "LLM Glossary — the terms that matter when comparing AI models",
    description: "Context windows, benchmarks, throughput, caching, quantization — defined in plain English.",
    url: "/glossary",
    type: "website",
  },
};

export default function GlossaryIndex() {
  return (
    <div className="min-h-screen">
      <div className="mx-auto w-full max-w-[1200px] px-5 py-7">
        <SiteHeader />

        <h1 className="font-display mt-8 text-[30px] font-bold text-ink sm:text-[34px]">LLM glossary</h1>
        <p className="mt-2 max-w-2xl text-[15px] leading-relaxed text-ink-2">
          The terms that matter when you compare AI models — defined in plain English, with the data to back them up.
        </p>

        <div className="mt-7 grid gap-3 sm:grid-cols-2">
          {GLOSSARY.map((t) => (
            <Link
              key={t.slug}
              href={`/glossary/${t.slug}`}
              className="card-lift group rounded-lg border border-line bg-surface p-4"
            >
              <h2 className="text-[15px] font-semibold text-ink group-hover:text-brand-ink">{t.term}</h2>
              <p className="mt-1.5 text-[13px] leading-relaxed text-ink-2">{t.short}</p>
            </Link>
          ))}
        </div>
      </div>
      <Footer />
    </div>
  );
}
