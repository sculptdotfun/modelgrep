import type { MetadataRoute } from "next";

export default function robots(): MetadataRoute.Robots {
  return {
    rules: { userAgent: "*", allow: "/" },
    sitemap: "https://modelgrep.com/sitemap.xml",
    host: "https://modelgrep.com",
  };
}
