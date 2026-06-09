import type { Metadata } from "next";
import Link from "next/link";
import { BLOG_POSTS } from "@/lib/blog";
import { Footer } from "@/components/Footer";
import { SiteHeader } from "@/components/SiteHeader";

export const metadata: Metadata = {
  title: "Blog — Practical Guides for Working with LLMs",
  description:
    "Practical guides for choosing, pricing, and deploying AI models: model selection frameworks, token pricing, coding models, speed vs cost trade-offs, and more.",
  alternates: { canonical: "/blog" },
  openGraph: {
    title: "modelgrep blog — practical guides for working with LLMs",
    description: "Model selection, pricing, speed and capability guides backed by live benchmark data.",
    url: "/blog",
    type: "website",
  },
};

export default function BlogIndex() {
  const posts = [...BLOG_POSTS].sort((a, b) => b.date.localeCompare(a.date));
  return (
    <div className="min-h-screen">
      <div className="mx-auto w-full max-w-[860px] px-5 py-7">
        <SiteHeader />

        <h1 className="font-display mt-8 text-[32px] font-bold text-ink">Blog</h1>
        <p className="mt-2 text-[15px] text-ink-2">Practical guides for working with AI models.</p>

        <div className="mt-7 grid gap-3 sm:grid-cols-2">
          {posts.map((p) => (
            <Link
              key={p.slug}
              href={`/blog/${p.slug}`}
              className="card-shadow card-lift group flex flex-col rounded-lg border border-line bg-surface p-5"
            >
              <span className="self-start font-mono text-[10px] font-semibold uppercase tracking-widest text-ink-3">
                {p.tag}
              </span>
              <h2 className="font-display mt-3 text-[17px] font-bold leading-snug text-ink group-hover:text-brand-ink">
                {p.title}
              </h2>
              <p className="mt-2 text-[13px] leading-relaxed text-ink-2">{p.excerpt}</p>
              <span className="mt-auto pt-3 text-[12px] font-medium text-ink-3">Read →</span>
            </Link>
          ))}
        </div>
      </div>
      <Footer />
    </div>
  );
}
