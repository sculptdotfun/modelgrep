// GET /api/v1/makers — every model maker with counts and its headline models.
// Mirrors the /makers pages; handy for building maker filters against /models.

import { getCatalog } from "@/lib/catalog";
import { groupByMaker } from "@/lib/makers";
import { apiHeaders, json, apiError, apiMaker } from "@/lib/api";

export const revalidate = 3600;

export function OPTIONS() {
  return new Response(null, { status: 204, headers: apiHeaders });
}

export async function GET() {
  let catalog;
  try {
    catalog = await getCatalog();
  } catch {
    return apiError(503, "Catalog temporarily unavailable — try again shortly.");
  }

  const makers = groupByMaker(catalog).map(apiMaker);
  return json({ data: makers, meta: { count: makers.length, generated_at: new Date().toISOString() } });
}
