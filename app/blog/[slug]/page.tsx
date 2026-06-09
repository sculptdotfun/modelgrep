import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { BLOG_POSTS, getPost } from "@/lib/blog";
import { Footer } from "@/components/Footer";
import { SiteHeader } from "@/components/SiteHeader";

type Params = { slug: string };

export function generateStaticParams() {
  return BLOG_POSTS.map((p) => ({ slug: p.slug }));
}

export async function generateMetadata({ params }: { params: Promise<Params> }): Promise<Metadata> {
  const { slug } = await params;
  const post = getPost(slug);
  if (!post) return { title: "Post not found" };
  return {
    title: post.title,
    description: post.excerpt,
    alternates: { canonical: `/blog/${post.slug}` },
    openGraph: { title: post.title, description: post.excerpt, url: `/blog/${post.slug}`, type: "article" },
    twitter: { card: "summary_large_image", title: post.title, description: post.excerpt },
  };
}

export default async function BlogPost({ params }: { params: Promise<Params> }) {
  const { slug } = await params;
  const post = getPost(slug);
  if (!post) notFound();

  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "Article",
    headline: post.title,
    description: post.excerpt,
    datePublished: post.date,
    author: { "@type": "Organization", name: "modelgrep", url: "https://modelgrep.com" },
    publisher: { "@type": "Organization", name: "modelgrep" },
    mainEntityOfPage: `https://modelgrep.com/blog/${post.slug}`,
  };

  const others = BLOG_POSTS.filter((p) => p.slug !== post.slug).slice(0, 3);

  return (
    <div className="min-h-screen">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }} />
      <div className="mx-auto w-full max-w-[760px] px-5 py-7">
        <SiteHeader />

        <nav className="mt-7 text-xs text-ink-3">
          <Link href="/blog" className="hover:text-ink-2">
            blog
          </Link>
          <span className="mx-1.5">/</span>
          <span className="text-ink-2">{post.slug}</span>
        </nav>

        <span className="mt-5 inline-block rounded-full bg-brand/10 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-brand-ink">
          {post.tag}
        </span>
        <h1 className="font-display mt-3 text-[30px] font-bold leading-tight text-ink sm:text-[34px]">{post.title}</h1>
        <p className="mt-2 text-[15px] text-ink-2">{post.excerpt}</p>

        <article className="prose-mg mt-8" dangerouslySetInnerHTML={{ __html: post.html }} />

        <div className="mt-12 border-t border-line pt-6">
          <div className="mb-3 text-[11px] font-semibold uppercase tracking-wider text-ink-3">More guides</div>
          <div className="grid gap-3 sm:grid-cols-3">
            {others.map((p) => (
              <Link
                key={p.slug}
                href={`/blog/${p.slug}`}
                className="card-shadow card-lift rounded-xl border border-line bg-surface p-3.5 text-[13px] font-medium leading-snug text-ink hover:text-brand-ink"
              >
                {p.title}
              </Link>
            ))}
          </div>
        </div>
      </div>
      <Footer />
    </div>
  );
}
