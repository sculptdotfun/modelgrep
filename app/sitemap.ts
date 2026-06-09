import type { MetadataRoute } from "next";
import { getCatalog } from "@/lib/catalog";

export const revalidate = 3600;

const BASE = "https://modelgrep.com";

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const models = await getCatalog();
  const modelUrls: MetadataRoute.Sitemap = models.map((m) => ({
    url: `${BASE}/models/${m.id}`,
    changeFrequency: "daily",
    priority: m.aa?.intelligence != null ? 0.8 : 0.6,
  }));

  return [
    { url: BASE, changeFrequency: "hourly", priority: 1 },
    ...modelUrls,
  ];
}
