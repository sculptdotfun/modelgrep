// GET /api/v1/rankings/{collection}[/{maker}] — the "best LLM for X" ranking,
// reusing the same resolver that powers the /best pages so API and site agree.

import { getCatalog } from "@/lib/catalog";
import { resolveFacet } from "@/lib/facets";
import { apiHeaders, json, apiError, apiModel } from "@/lib/api";

export const revalidate = 3600;

export function OPTIONS() {
  return new Response(null, { status: 204, headers: apiHeaders });
}

export async function GET(_req: Request, { params }: { params: Promise<{ slug: string[] }> }) {
  const { slug } = await params;

  let catalog;
  try {
    catalog = await getCatalog();
  } catch {
    return apiError(503, "Catalog temporarily unavailable — try again shortly.");
  }

  const facet = resolveFacet(slug, catalog);
  if (!facet) {
    return apiError(404, `Unknown ranking "${slug.join("/")}". See GET /api/v1 for valid collections.`);
  }

  return json({
    slug: facet.base.slug,
    title: facet.h1,
    metric: facet.base.metricLabel,
    maker: facet.maker?.slug ?? null,
    answer: facet.answer,
    count: facet.models.length,
    data: facet.models.map((m, i) => ({
      rank: i + 1,
      metric_value: facet.base.value(m),
      metric_display: facet.base.display(m),
      ...apiModel(m),
    })),
  });
}
