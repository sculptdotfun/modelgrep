import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { GLOSSARY, getTerm } from "@/lib/glossary";
import { Footer } from "@/components/Footer";
import { SiteHeader } from "@/components/SiteHeader";

type Params = { slug: string };

export function generateStaticParams() {
  return GLOSSARY.map((t) => ({ slug: t.slug }));
}

export async function generateMetadata({ params }: { params: Promise<Params> }): Promise<Metadata> {
  const { slug } = await params;
  const t = getTerm(slug);
  if (!t) return { title: "Term not found" };
  const title = `What is ${t.term}? — LLM Glossary`;
  return {
    title,
    description: t.short,
    keywords: [t.term, `what is ${t.term.toLowerCase()}`, `${t.term} LLM`, `${t.term} meaning`, "LLM glossary"],
    alternates: { canonical: `/glossary/${t.slug}` },
    openGraph: { title, description: t.short, url: `/glossary/${t.slug}`, type: "article" },
    twitter: { card: "summary_large_image", title, description: t.short },
  };
}

export default async function GlossaryTermPage({ params }: { params: Promise<Params> }) {
  const { slug } = await params;
  const t = getTerm(slug);
  if (!t) notFound();

  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "DefinedTerm",
    name: t.term,
    description: t.short,
    url: `https://modelgrep.com/glossary/${t.slug}`,
    inDefinedTermSet: { "@type": "DefinedTermSet", name: "modelgrep LLM glossary", url: "https://modelgrep.com/glossary" },
  };

  const others = GLOSSARY.filter((x) => x.slug !== t.slug).slice(0, 6);

  return (
    <div className="min-h-screen">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }} />
      <div className="mx-auto w-full max-w-[760px] px-5 py-7">
        <SiteHeader />

        <nav className="mt-7 text-xs text-ink-3">
          <Link href="/glossary" className="hover:text-ink-2">
            glossary
          </Link>
          <span className="mx-1.5">/</span>
          <span className="text-ink-2">{t.slug}</span>
        </nav>

        <h1 className="font-display mt-4 text-[30px] font-bold leading-tight text-ink sm:text-[34px]">{t.term}</h1>
        <p className="mt-3 text-[16px] font-medium leading-relaxed text-ink">{t.short}</p>

        <div className="mt-6 space-y-4">
          {t.body.map((p, i) => (
            <p key={i} className="text-[15.5px] leading-[1.75] text-ink-2">
              {p}
            </p>
          ))}
        </div>

        {t.related.length > 0 && (
          <div className="mt-8 flex flex-wrap gap-2">
            {t.related.map((r) => (
              <Link
                key={r.href}
                href={r.href}
                className="rounded-md border border-line bg-surface px-3 py-1.5 text-[13px] font-medium text-ink-2 transition-colors hover:border-line-strong hover:text-ink"
              >
                {r.label} →
              </Link>
            ))}
          </div>
        )}

        <div className="mt-12 border-t border-line pt-6">
          <div className="mb-3 text-[11px] font-semibold uppercase tracking-wider text-ink-3">More terms</div>
          <div className="grid gap-2 sm:grid-cols-2">
            {others.map((x) => (
              <Link key={x.slug} href={`/glossary/${x.slug}`} className="text-[13px] font-medium text-ink-2 hover:text-brand-ink">
                {x.term} →
              </Link>
            ))}
          </div>
        </div>
      </div>
      <Footer />
    </div>
  );
}
