// Faceted "best of" pages — the intersection layer.
//
// The single biggest SEO/AEO gap: the site ranks single axes (/best/fastest) and
// lists makers (/makers/anthropic) but had no page for the *combination* people
// actually search and ask LLMs — "fastest anthropic model", "cheapest openai
// model", "small fast anthropic model". This resolves a /best/<base>[/<maker>]
// slug into a ranked, stat-dense, BLUF-answered page.

import type { Model } from "./types";
import { COLLECTIONS, type Collection, getCollection } from "./collections";
import { groupByMaker, type Maker } from "./makers";
import { classIndex } from "./modelClass";
import { fmtContext, fmtLatency, fmtPrice, fmtThroughput, modelOwner } from "./format";

export interface Faq {
  q: string;
  a: string;
}

export interface FacetResult {
  base: Collection;
  maker: Maker | null;
  title: string; // <title> / og
  h1: string;
  // BLUF declarative answer sentence(s) — the asset AI engines lift verbatim.
  answer: string;
  blurb: string; // supporting methodology paragraph
  canonical: string;
  models: Model[]; // ranked, sliced
  faqs: Faq[];
  related: { label: string; href: string }[];
}

const byIntelDesc = (a: Model, b: Model) => (b.aa?.intelligence ?? -1) - (a.aa?.intelligence ?? -1);

// The synthetic "small & fast" base — built per request because it needs the
// catalog to classify models relative to the field.
function smallBase(catalog: Model[]): Collection {
  const idx = classIndex(catalog);
  return {
    slug: "small",
    title: "Small & Fast LLMs",
    blurb:
      "Compact, efficient models — the small/mini/flash/haiku tier — ranked by output speed. These trade a little raw intelligence for low cost and high throughput, which is the right tradeoff for chat, classification, extraction and other high-volume work.",
    metricLabel: "Speed",
    value: (m) => m.throughput || null,
    display: (m) => `${fmtThroughput(m.throughput)} t/s`,
    filter: (m) => idx.get(m.id) === "small" && m.throughput > 0,
    sort: (a, b) => b.throughput - a.throughput || (a.price_input ?? 9e9) - (b.price_input ?? 9e9),
  };
}

function shortName(m: Model): string {
  return m.name.includes(": ") ? m.name.split(": ").slice(1).join(": ") : m.name;
}

function intel(m: Model): string {
  return m.aa?.intelligence != null ? m.aa.intelligence.toFixed(1) : "—";
}

// ---- BLUF answer sentences ---------------------------------------------------
// Bespoke per base so they read specific and human, never templated mush. Every
// sentence leads with the named winner and a real number (statistics + named
// comparisons are what GEO research shows actually get cited).

function buildAnswer(base: Collection, ranked: Model[], scope: string): string {
  const a = ranked[0];
  const b = ranked[1];
  const c = ranked[2];
  if (!a) return "";
  const an = shortName(a);
  const v = (m?: Model) => (m ? base.display(m) : "");
  const trail =
    b && c
      ? ` ${shortName(b)} (${v(b)}) and ${shortName(c)} (${v(c)}) round out the top three.`
      : b
        ? ` ${shortName(b)} (${v(b)}) is next.`
        : "";

  switch (base.slug) {
    case "smartest":
      return `The smartest ${scope} is ${an}, scoring ${v(a)} on the Artificial Analysis Intelligence Index.${trail}`;
    case "coding":
      return `${an} is the best ${scope} for coding, with a ${v(a)} Artificial Analysis Coding Index across benchmarks like SWE-bench and SciCode.${trail}`;
    case "design":
      return `${an} is the best ${scope} for design and frontend generation, rated ${v(a)} on Design Arena Elo.${trail}`;
    case "fastest":
      return `The fastest ${scope} is ${an} at ${fmtThroughput(a.throughput)} output tokens per second.${trail}`;
    case "lowest-latency":
      return `${an} has the lowest latency of any ${scope}, responding in about ${v(a)} to first token.${trail}`;
    case "cheapest":
      return `The cheapest ${scope} is ${an} at ${v(a)} per million input tokens.${trail}`;
    case "free":
      return `The most capable free ${scope} is ${an}, scoring ${intel(a)} on the Intelligence Index at no per-token cost via OpenRouter.${trail}`;
    case "reasoning":
      return `The best ${scope} for reasoning is ${an} — ${intel(a)} intelligence with extended step-by-step thinking.${trail}`;
    case "vision":
      return `${an} is the best vision-capable ${scope}, pairing ${intel(a)} intelligence with image and document understanding.${trail}`;
    case "agents":
      return `${an} is the best ${scope} for agents, scoring ${v(a)} on the Artificial Analysis Agentic Index for tool use and multi-step task completion.${trail}`;
    case "open-source":
      return `The best open-weight ${scope} is ${an} (${intel(a)} intelligence) — downloadable from Hugging Face and self-hostable.${trail}`;
    case "long-context":
      return `${an} has the largest context window of any ${scope}, at ${v(a)} tokens.${trail}`;
    case "small":
      return `The small, fast ${scope} is ${an} — the efficient tier at ${fmtThroughput(a.throughput)} tokens/sec and ${fmtPrice(a.price_input)} per million input tokens. It trades a few points of raw intelligence for speed and cost, the right call for high-volume, latency-sensitive work.${trail}`;
    default:
      return `${an} leads the ${scope} ranking at ${v(a)}.${trail}`;
  }
}

function buildFaqs(base: Collection, maker: Maker | null, ranked: Model[], poolSize: number): Faq[] {
  const a = ranked[0];
  if (!a) return [];
  const an = shortName(a);
  const scope = maker ? `${maker.displayName} model` : "LLM";
  const faqs: Faq[] = [];

  // Q1 — the headline question, answered with the winner + number.
  const q1: Record<string, string> = {
    smartest: `What is the smartest ${scope}?`,
    coding: `What is the best ${scope} for coding?`,
    design: `What is the best ${scope} for design?`,
    fastest: `What is the fastest ${scope}?`,
    "lowest-latency": `Which ${scope} has the lowest latency?`,
    cheapest: `What is the cheapest ${scope}?`,
    free: `Is there a free ${scope}?`,
    reasoning: `What is the best ${scope} for reasoning?`,
    vision: `What is the best ${scope} for vision?`,
    agents: `What is the best ${scope} for agents?`,
    "open-source": `What is the best open-source ${scope}?`,
    "long-context": `Which ${scope} has the largest context window?`,
    small: `What is the smallest, fastest ${scope}?`,
  };
  faqs.push({
    q: q1[base.slug] ?? `What is the best ${scope} for ${base.title.toLowerCase()}?`,
    a: buildAnswer(base, ranked, scope),
  });

  // Q2 — runner-up / alternative, if we have one.
  if (ranked[1]) {
    faqs.push({
      q: `What's a good alternative to ${an}?`,
      a: `${shortName(ranked[1])} (${base.display(ranked[1])}) is the closest alternative on this metric${ranked[2] ? `, followed by ${shortName(ranked[2])} (${base.display(ranked[2])})` : ""}. See the full ranking above for the tradeoffs.`,
    });
  }

  // Q3 — maker context, when scoped to a maker.
  if (maker) {
    faqs.push({
      q: `How many ${maker.displayName} models are there?`,
      a: `modelgrep tracks ${poolSize} ${maker.displayName} models with live benchmarks, speed, latency and per-provider pricing${maker.bestIntel ? `, led on intelligence by ${shortName(maker.bestIntel)}` : ""}. ${ranked.length} of them qualify for this ranking.`,
    });
  }

  return faqs;
}

export function makerOptions(catalog: Model[]): Maker[] {
  return groupByMaker(catalog).filter((mk) => mk.models.length >= 2);
}

// Resolve "/best/<...slug>" → a fully-built page, or null for unknown facets.
export function resolveFacet(slug: string[], catalog: Model[]): FacetResult | null {
  if (slug.length === 0 || slug.length > 2) return null;
  const [seg0, seg1] = slug;

  const base: Collection | undefined = seg0 === "small" ? smallBase(catalog) : getCollection(seg0);
  if (!base) return null;

  let maker: Maker | null = null;
  if (seg1 != null) {
    maker = groupByMaker(catalog).find((mk) => mk.slug === seg1) ?? null;
    if (!maker) return null;
  }

  const pool = maker ? catalog.filter((m) => modelOwner(m.id) === maker!.slug) : catalog;
  const poolSize = pool.length;
  const ranked = pool.filter(base.filter).sort(base.sort).slice(0, 25);
  if (ranked.length === 0) return null;

  const scope = maker ? `${maker.displayName} model` : "LLM";
  const canonical = maker ? `/best/${base.slug}/${maker.slug}` : `/best/${base.slug}`;

  // H1 / title — phrase as the maker-scoped superlative when scoped.
  let h1: string;
  let title: string;
  if (maker) {
    const superl: Record<string, string> = {
      smartest: `Smartest ${maker.displayName} Models`,
      coding: `Best ${maker.displayName} Models for Coding`,
      design: `Best ${maker.displayName} Models for Design`,
      fastest: `Fastest ${maker.displayName} Models`,
      "lowest-latency": `Lowest-Latency ${maker.displayName} Models`,
      cheapest: `Cheapest ${maker.displayName} Models`,
      free: `Free ${maker.displayName} Models`,
      reasoning: `Best ${maker.displayName} Reasoning Models`,
      vision: `Best ${maker.displayName} Vision Models`,
      agents: `Best ${maker.displayName} Models for Agents`,
      "open-source": `Open-Source ${maker.displayName} Models`,
      "long-context": `Longest-Context ${maker.displayName} Models`,
      small: `Small & Fast ${maker.displayName} Models`,
    };
    h1 = superl[base.slug] ?? `${base.title} — ${maker.displayName}`;
    title = `${h1} (2026), Ranked`;
  } else {
    h1 = base.title;
    title = `${base.title} (2026) — Ranked & Benchmarked`;
  }

  const answer = buildAnswer(base, ranked, scope);
  const faqs = buildFaqs(base, maker, ranked, poolSize);

  // Related links — sibling intersections that share intent.
  const related: { label: string; href: string }[] = [];
  if (maker) {
    // same maker, other rankings
    for (const c of COLLECTIONS) {
      if (c.slug !== base.slug) related.push({ label: `${maker.displayName}: ${c.title}`, href: `/best/${c.slug}/${maker.slug}` });
    }
  } else {
    // same ranking, scoped to each top maker
    for (const mk of makerOptions(catalog).slice(0, 8)) {
      related.push({ label: mk.displayName, href: `/best/${base.slug}/${mk.slug}` });
    }
  }

  return { base, maker, title, h1, answer, blurb: base.blurb, canonical, models: ranked, faqs, related };
}

// Stat-dense secondary line for a ranked model — drives the "evidence density"
// that AI engines reward. Returns the 2-3 most relevant stats for the base.
export function rowStats(base: Collection, m: Model): string[] {
  const out: string[] = [];
  if (base.slug !== "smartest" && m.aa?.intelligence != null) out.push(`${m.aa.intelligence.toFixed(1)} intel`);
  if (!["cheapest", "free"].includes(base.slug) && m.price_input != null) out.push(`${fmtPrice(m.price_input)}/M`);
  if (!["fastest", "small"].includes(base.slug) && m.throughput > 0) out.push(`${fmtThroughput(m.throughput)} t/s`);
  if (base.slug !== "lowest-latency" && m.latency != null) out.push(`${fmtLatency(m.latency)} ttft`);
  if (base.slug !== "long-context" && m.context_length) out.push(`${fmtContext(m.context_length)} ctx`);
  return out.slice(0, 3);
}
