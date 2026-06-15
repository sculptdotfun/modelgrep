// GET /api/v1/models — list, filter, sort and paginate the catalog.

import { getCatalog } from "@/lib/catalog";
import { apiHeaders, json, apiError, parseQuery, queryModels } from "@/lib/api";

export const revalidate = 3600;

export function OPTIONS() {
  return new Response(null, { status: 204, headers: apiHeaders });
}

export async function GET(req: Request) {
  let catalog;
  try {
    catalog = await getCatalog();
  } catch {
    return apiError(503, "Catalog temporarily unavailable — try again shortly.");
  }

  const url = new URL(req.url);
  const params = parseQuery(url);
  const { data, total } = queryModels(catalog, params);

  const end = params.offset + data.length;
  const has_more = end < total;

  return json({
    data,
    meta: {
      total,
      count: data.length,
      limit: params.limit,
      offset: params.offset,
      has_more,
      next_offset: has_more ? end : null,
      generated_at: new Date().toISOString(),
    },
  });
}
