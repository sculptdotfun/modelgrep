// GET /api/v1/models/{id} — single model with provider breakdown. The id is a
// catch-all because model ids contain a slash (e.g. anthropic/claude-sonnet-4.5).

import { getModelDetail } from "@/lib/catalog";
import { apiHeaders, json, apiError, apiModelDetail } from "@/lib/api";

export const revalidate = 3600;

export function OPTIONS() {
  return new Response(null, { status: 204, headers: apiHeaders });
}

export async function GET(_req: Request, { params }: { params: Promise<{ id: string[] }> }) {
  const { id } = await params;
  const modelId = id.join("/");

  let detail;
  try {
    detail = await getModelDetail(modelId);
  } catch {
    return apiError(503, "Catalog temporarily unavailable — try again shortly.");
  }
  if (!detail) {
    return apiError(404, `No model with id "${modelId}". See GET /api/v1/models for valid ids.`);
  }

  return json({ data: apiModelDetail(detail) });
}
