// Group the catalog by model maker (the org before the slash in the id) for
// /makers pages — "anthropic models", "openai models" search intent.
import type { Model } from "./types";
import { modelOwner } from "./format";

export interface Maker {
  slug: string;
  displayName: string;
  models: Model[];
  bestIntel: Model | null;
  cheapest: Model | null;
  fastest: Model | null;
}

export function groupByMaker(models: Model[]): Maker[] {
  const bySlug = new Map<string, Model[]>();
  for (const m of models) {
    const owner = modelOwner(m.id);
    if (!bySlug.has(owner)) bySlug.set(owner, []);
    bySlug.get(owner)!.push(m);
  }

  const makers: Maker[] = [];
  for (const [slug, list] of bySlug) {
    // Display name from the "Owner: Model" name prefix when present.
    const named = list.find((m) => m.name.includes(": "));
    const displayName = named ? named.name.split(": ")[0] : slug.charAt(0).toUpperCase() + slug.slice(1);

    const withIntel = list.filter((m) => m.aa?.intelligence != null);
    const bestIntel = withIntel.length
      ? withIntel.reduce((a, b) => (b.aa!.intelligence! > a.aa!.intelligence! ? b : a))
      : null;
    const paid = list.filter((m) => (m.price_input ?? 0) > 0);
    const cheapest = paid.length ? paid.reduce((a, b) => (b.price_input! < a.price_input! ? b : a)) : null;
    const fast = list.filter((m) => m.throughput > 0);
    const fastest = fast.length ? fast.reduce((a, b) => (b.throughput > a.throughput ? b : a)) : null;

    makers.push({
      slug,
      displayName,
      models: [...list].sort(
        (a, b) => (b.aa?.intelligence ?? -1) - (a.aa?.intelligence ?? -1) || (b.created ?? 0) - (a.created ?? 0),
      ),
      bestIntel,
      cheapest,
      fastest,
    });
  }

  return makers.sort((a, b) => b.models.length - a.models.length);
}
