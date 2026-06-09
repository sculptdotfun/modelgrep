import type { MetadataRoute } from "next";
import { getCatalog } from "@/lib/catalog";
import { COLLECTIONS } from "@/lib/collections";
import { BLOG_POSTS } from "@/lib/blog";

export const revalidate = 3600;

const BASE = "https://modelgrep.com";

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const models = await getCatalog();
  const now = new Date();

  const modelUrls: MetadataRoute.Sitemap = models.map((m) => ({
    url: `${BASE}/models/${m.id}`,
    lastModified: now,
    changeFrequency: "daily",
    priority: m.aa?.intelligence != null ? 0.8 : 0.6,
  }));

  const collectionUrls: MetadataRoute.Sitemap = COLLECTIONS.map((c) => ({
    url: `${BASE}/best/${c.slug}`,
    lastModified: now,
    changeFrequency: "daily",
    priority: 0.7,
  }));

  // Pre-render-worthy comparison pages: all pairs among the top models.
  const top = models
    .filter((m) => m.aa?.intelligence != null)
    .sort((a, b) => b.aa!.intelligence! - a.aa!.intelligence!)
    .slice(0, 14);
  const compareUrls: MetadataRoute.Sitemap = [];
  for (let i = 0; i < top.length; i++) {
    for (let j = i + 1; j < top.length; j++) {
      const [a, b] = [top[i].id, top[j].id].sort();
      compareUrls.push({
        url: `${BASE}/compare/${a}/vs/${b}`,
        lastModified: now,
        changeFrequency: "weekly",
        priority: 0.6,
      });
    }
  }

  const blogUrls: MetadataRoute.Sitemap = [
    { url: `${BASE}/blog`, lastModified: now, changeFrequency: "weekly", priority: 0.6 },
    ...BLOG_POSTS.map((p) => ({
      url: `${BASE}/blog/${p.slug}`,
      lastModified: new Date(p.date),
      changeFrequency: "monthly" as const,
      priority: 0.5,
    })),
  ];

  return [
    { url: BASE, lastModified: now, changeFrequency: "hourly", priority: 1 },
    ...collectionUrls,
    ...modelUrls,
    ...compareUrls,
    ...blogUrls,
  ];
}
