// "Best LLMs for X" collection landing pages — programmatic, data-backed.
import type { Model } from "./types";
import { fmtContext, fmtLatency, fmtPrice, fmtThroughput } from "./format";

export interface Collection {
  slug: string;
  title: string; // <title> / h1
  blurb: string; // SEO intro paragraph
  metricLabel: string;
  value: (m: Model) => number | null;
  display: (m: Model) => string;
  filter: (m: Model) => boolean;
  // sort comparator (already-filtered models)
  sort: (a: Model, b: Model) => number;
}

const byIntelDesc = (a: Model, b: Model) => (b.aa?.intelligence ?? -1) - (a.aa?.intelligence ?? -1);

export const COLLECTIONS: Collection[] = [
  {
    slug: "smartest",
    title: "Smartest LLMs",
    blurb:
      "The most capable AI models ranked by the Artificial Analysis Intelligence Index — a composite of reasoning, knowledge, math and science benchmarks. These are the highest-scoring large language models available today.",
    metricLabel: "Intelligence",
    value: (m) => m.aa?.intelligence ?? null,
    display: (m) => m.aa!.intelligence!.toFixed(1),
    filter: (m) => m.aa?.intelligence != null,
    sort: byIntelDesc,
  },
  {
    slug: "coding",
    title: "Best LLMs for Coding",
    blurb:
      "AI models ranked by the Artificial Analysis Coding Index, measuring real-world software engineering ability across benchmarks like SWE-bench, SciCode and terminal tasks. The best LLMs for code generation, debugging and agentic development.",
    metricLabel: "Coding",
    value: (m) => m.aa?.coding ?? null,
    display: (m) => m.aa!.coding!.toFixed(1),
    filter: (m) => m.aa?.coding != null,
    sort: (a, b) => (b.aa?.coding ?? -1) - (a.aa?.coding ?? -1),
  },
  {
    slug: "design",
    title: "Best LLMs for Design & Frontend",
    blurb:
      "AI models ranked by Design Arena Elo — head-to-head human preference for generating UI, websites, data visualizations and 3D. The best models for frontend and design generation.",
    metricLabel: "Design Elo",
    value: (m) => m.da?.elo ?? null,
    display: (m) => String(m.da!.elo),
    filter: (m) => m.da?.elo != null,
    sort: (a, b) => (b.da?.elo ?? -1) - (a.da?.elo ?? -1),
  },
  {
    slug: "fastest",
    title: "Fastest LLMs",
    blurb:
      "AI models ranked by output speed (tokens per second, p50). The fastest large language models for low-latency and high-throughput applications.",
    metricLabel: "Speed",
    value: (m) => m.throughput || null,
    display: (m) => `${fmtThroughput(m.throughput)} t/s`,
    filter: (m) => m.throughput > 0,
    sort: (a, b) => b.throughput - a.throughput,
  },
  {
    slug: "lowest-latency",
    title: "Lowest-Latency LLMs",
    blurb:
      "AI models ranked by time-to-first-token (p50). The most responsive large language models for real-time and interactive use cases.",
    metricLabel: "Latency",
    value: (m) => m.latency,
    display: (m) => fmtLatency(m.latency),
    filter: (m) => m.latency != null,
    sort: (a, b) => (a.latency ?? Infinity) - (b.latency ?? Infinity),
  },
  {
    slug: "cheapest",
    title: "Cheapest LLMs",
    blurb:
      "AI models ranked by input token price. The most affordable large language model APIs, from budget open-weight models to discounted frontier models.",
    metricLabel: "Input /M",
    value: (m) => m.price_input,
    display: (m) => fmtPrice(m.price_input),
    filter: (m) => (m.price_input ?? 0) > 0,
    sort: (a, b) => (a.price_input ?? Infinity) - (b.price_input ?? Infinity),
  },
  {
    slug: "free",
    title: "Best Free LLMs",
    blurb:
      "The best large language models with a free tier, ranked by intelligence. Capable AI models you can use at no cost via OpenRouter.",
    metricLabel: "Intelligence",
    value: (m) => m.aa?.intelligence ?? null,
    display: (m) => (m.aa?.intelligence != null ? m.aa.intelligence.toFixed(1) : "—"),
    filter: (m) => (m.price_input ?? 1) === 0,
    sort: byIntelDesc,
  },
  {
    slug: "reasoning",
    title: "Best Reasoning LLMs",
    blurb:
      "Large language models with extended reasoning / thinking, ranked by intelligence. The best models for complex multi-step reasoning, math and analysis.",
    metricLabel: "Intelligence",
    value: (m) => m.aa?.intelligence ?? null,
    display: (m) => (m.aa?.intelligence != null ? m.aa.intelligence.toFixed(1) : "—"),
    filter: (m) => m.capabilities.reasoning,
    sort: byIntelDesc,
  },
  {
    slug: "vision",
    title: "Best Vision LLMs",
    blurb:
      "Multimodal large language models that accept image input, ranked by intelligence. The best vision-capable AI models for understanding images, documents and charts.",
    metricLabel: "Intelligence",
    value: (m) => m.aa?.intelligence ?? null,
    display: (m) => (m.aa?.intelligence != null ? m.aa.intelligence.toFixed(1) : "—"),
    filter: (m) => m.capabilities.vision,
    sort: byIntelDesc,
  },
  {
    slug: "long-context",
    title: "Longest-Context LLMs",
    blurb:
      "AI models with the largest context windows, ranked by token capacity. The best large language models for long documents, codebases and extended conversations.",
    metricLabel: "Context",
    value: (m) => m.context_length || null,
    display: (m) => fmtContext(m.context_length),
    filter: (m) => (m.context_length ?? 0) >= 200_000,
    sort: (a, b) => (b.context_length ?? 0) - (a.context_length ?? 0),
  },
];

export function getCollection(slug: string): Collection | undefined {
  return COLLECTIONS.find((c) => c.slug === slug);
}
