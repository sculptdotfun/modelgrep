import type { Metadata } from "next";
import Link from "next/link";
import { AnswerBox } from "@/components/AnswerBox";
import { Footer } from "@/components/Footer";
import { COLLECTIONS } from "@/lib/collections";
import { fmtMonth } from "@/lib/format";
import { CAPABILITIES, SITE, SORT_FIELDS } from "@/lib/api";

export const revalidate = 3600;

const TITLE = "Free LLM API — Benchmarks, Pricing & Speed for 300+ Models";
const DESCRIPTION =
  "A free, no-key JSON API for 300+ large language models. Query AI models by benchmark score (Artificial Analysis, Design Arena Elo), output speed, latency, price and capability. CORS-enabled, no signup.";

export const metadata: Metadata = {
  title: TITLE,
  description: DESCRIPTION,
  keywords: [
    "free LLM API",
    "AI model API",
    "LLM benchmark API",
    "compare AI models API",
    "LLM pricing API",
    "OpenRouter models API",
    "AI model data JSON",
    "best LLM API",
  ],
  alternates: { canonical: "/api" },
  openGraph: { title: TITLE, description: DESCRIPTION, url: "/api", type: "article" },
  twitter: { card: "summary_large_image", title: TITLE, description: DESCRIPTION },
};

// The BLUF answer — the asset answer engines lift verbatim.
const ANSWER =
  "modelgrep's API is a free, no-key JSON endpoint for querying 300+ large language models by benchmark score, output speed, latency, price and capability. Call GET https://modelgrep.com/api/v1/models — no signup, CORS-enabled, and the same data that powers the site's rankings.";

// FAQs do double duty: visible accordion content + FAQPage structured data.
const FAQS: { q: string; a: string }[] = [
  {
    q: "Is there a free LLM API?",
    a: "Yes. The modelgrep API is free and requires no API key or signup. It exposes benchmarks, live speed and latency, pricing and capabilities for 300+ models. Responses are cached for about an hour; please keep request volume reasonable.",
  },
  {
    q: "How do I get LLM benchmark data through an API?",
    a: "Call GET https://modelgrep.com/api/v1/models. Each model includes its Artificial Analysis Intelligence, Coding and Agentic index scores plus Design Arena Elo under the benchmarks field. Sort by any of them with ?sort=intelligence (or coding, agentic, design).",
  },
  {
    q: "Does the modelgrep API require an API key or authentication?",
    a: "No. There is no key, token or signup. CORS is open (Access-Control-Allow-Origin: *), so you can call it directly from a browser, a serverless function or a script.",
  },
  {
    q: "How do I find the cheapest or fastest LLM programmatically?",
    a: "Use the sort parameter — ?sort=price_input for the cheapest input pricing, or ?sort=throughput for the fastest output. You can also hit the ranked endpoints GET /api/v1/rankings/cheapest and /api/v1/rankings/fastest, which return a pre-ranked list and a one-sentence answer.",
  },
  {
    q: "Can I compare AI model prices and context windows via the API?",
    a: "Yes. Every model returns a pricing object (input, output and cache-read in USD per million tokens) and context_length in tokens, alongside capabilities and per-provider pricing on the single-model endpoint.",
  },
  {
    q: "What are the rate limits?",
    a: "There is no hard rate limit, but it is a free service on fair-use terms. Responses are cached roughly an hour at the edge, so repeated identical requests are cheap. Cache aggressively on your side rather than polling.",
  },
  {
    q: "Is this the official OpenRouter API?",
    a: "No. modelgrep is an independent project, not affiliated with OpenRouter. It aggregates and enriches public data from OpenRouter, Artificial Analysis and Design Arena into one consistent, read-only JSON shape.",
  },
];

// Quickstart samples kept inline (server-rendered) so the code is in the HTML
// for search engines — no client-side tabs.
const SAMPLES: { lang: string; code: string }[] = [
  {
    lang: "curl",
    code: `curl "${SITE}/api/v1/models?sort=intelligence&limit=5"`,
  },
  {
    lang: "JavaScript",
    code: `const res = await fetch("${SITE}/api/v1/models?sort=coding&limit=5");
const { data } = await res.json();
console.log(data.map((m) => m.id));`,
  },
  {
    lang: "Python",
    code: `import requests
r = requests.get("${SITE}/api/v1/models", params={"sort": "price_input", "free": 1})
for m in r.json()["data"]:
    print(m["id"], m["pricing"]["input"])`,
  },
];

function Code({ children }: { children: React.ReactNode }) {
  return (
    <pre className="card-shadow overflow-x-auto rounded-lg border border-line bg-surface-2 p-3.5 font-mono text-[12.5px] leading-relaxed text-ink-2">
      {children}
    </pre>
  );
}

function Endpoint({ method, path, children }: { method: string; path: string; children: React.ReactNode }) {
  return (
    <div className="mt-6">
      <div className="flex items-center gap-2.5">
        <span className="rounded border border-line bg-surface-2 px-1.5 py-0.5 font-mono text-[11px] font-semibold text-ink">
          {method}
        </span>
        <code className="font-mono text-[13.5px] font-semibold text-ink">{path}</code>
      </div>
      <div className="mt-2.5 text-[14px] leading-relaxed text-ink-2">{children}</div>
    </div>
  );
}

function Row({ name, desc }: { name: string; desc: string }) {
  return (
    <div className="grid grid-cols-[140px_1fr] gap-3 border-t border-line py-2 text-[13px]">
      <code className="font-mono text-ink">{name}</code>
      <span className="text-ink-2">{desc}</span>
    </div>
  );
}

export default function ApiDocs() {
  const updated = fmtMonth(new Date());

  const jsonLd = [
    {
      "@context": "https://schema.org",
      "@type": "WebAPI",
      name: "modelgrep API",
      description: DESCRIPTION,
      url: `${SITE}/api`,
      documentation: `${SITE}/api`,
      provider: { "@type": "Organization", name: "modelgrep", url: SITE },
      offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
      isAccessibleForFree: true,
    },
    {
      "@context": "https://schema.org",
      "@type": "BreadcrumbList",
      itemListElement: [
        { "@type": "ListItem", position: 1, name: "modelgrep", item: SITE },
        { "@type": "ListItem", position: 2, name: "API", item: `${SITE}/api` },
      ],
    },
    {
      "@context": "https://schema.org",
      "@type": "FAQPage",
      mainEntity: FAQS.map((f) => ({
        "@type": "Question",
        name: f.q,
        acceptedAnswer: { "@type": "Answer", text: f.a },
      })),
    },
  ];

  return (
    <div className="min-h-screen">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }} />
      <div className="mx-auto w-full max-w-[820px] px-5 py-7">
        <nav className="mt-5 text-xs text-ink-3">
          <Link href="/" className="hover:text-ink-2">
            home
          </Link>
          <span className="mx-1.5">/</span>
          <span className="text-ink">api</span>
        </nav>

        <h1 className="font-display mt-4 text-[32px] font-bold leading-tight text-ink sm:text-[38px]">
          Free LLM API
        </h1>

        <AnswerBox answer={ANSWER} updated={updated} />

        <p className="mt-5 max-w-3xl text-[15px] leading-relaxed text-ink-2">
          One JSON endpoint for the whole LLM landscape — benchmark scores, live throughput and
          latency, token pricing and capabilities for 300+ models on OpenRouter. No API key, no
          signup, CORS-enabled. It serves the exact data behind modelgrep&apos;s{" "}
          <Link href="/best" className="text-ink underline underline-offset-2">
            rankings
          </Link>{" "}
          and{" "}
          <Link href="/compare" className="text-ink underline underline-offset-2">
            comparisons
          </Link>
          , so you can build your own model picker, leaderboard or cost report on top of it.
        </p>

        {/* Quickstart */}
        <h2 className="mt-10 font-display text-[20px] font-bold tracking-tight text-ink">Quickstart</h2>
        <p className="mt-2 text-[14px] text-ink-2">
          No setup. Hit the base URL <code className="font-mono text-[13px]">{`${SITE}/api/v1`}</code>{" "}
          and parse JSON.
        </p>
        {SAMPLES.map((s) => (
          <div key={s.lang} className="mt-4">
            <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-wider text-ink-3">{s.lang}</div>
            <Code>{s.code}</Code>
          </div>
        ))}

        {/* List */}
        <h2 className="mt-10 font-display text-[20px] font-bold tracking-tight text-ink">List models</h2>
        <Endpoint method="GET" path="/api/v1/models">
          Filter, sort and paginate the full catalog. All parameters are optional.
        </Endpoint>
        <div className="mt-3">
          <Row name="q" desc="Search id + name (space-separated terms, OR-matched)." />
          <Row name="maker" desc="Maker slug — anthropic, openai, google, meta-llama, …" />
          <Row name="provider" desc="Serving-provider substring." />
          <Row name="free" desc="1 → only $0 input-price models." />
          <Row name="benchmarked" desc="1 → only models that have benchmark data." />
          <Row name="capabilities" desc={`Comma list: ${CAPABILITIES.join(", ")}.`} />
          <Row name="max_price" desc="Max input price, USD per million tokens." />
          <Row name="min_context" desc="Minimum context window, in tokens." />
          <Row name="sort" desc={SORT_FIELDS.join(" · ")} />
          <Row name="order" desc="asc | desc (sensible default per field)." />
          <Row name="limit" desc="1–200 (default 50)." />
          <Row name="offset" desc="Pagination offset (default 0)." />
        </div>
        <p className="mt-5 text-[13px] font-semibold uppercase tracking-wider text-ink-3">Example response</p>
        <div className="mt-2">
          <Code>{`{
  "data": [
    {
      "id": "google/gemini-2.5-flash-lite",
      "name": "Google: Gemini 2.5 Flash Lite",
      "maker": "google",
      "context_length": 1048576,
      "pricing": { "input": 0.1, "output": 0.4, "unit": "usd_per_million_tokens" },
      "performance": { "throughput_tps": 320.5, "latency_ms": 410, "uptime": 99.9 },
      "capabilities": { "vision": true, "tools": true, "reasoning": false, ... },
      "benchmarks": { "artificial_analysis": { "intelligence": 46.2, ... }, "design_arena": null },
      "url": "${SITE}/models/google/gemini-2.5-flash-lite"
    }
  ],
  "meta": { "total": 23, "count": 5, "limit": 5, "offset": 0, "has_more": true, "next_offset": 5 }
}`}</Code>
        </div>

        {/* Detail */}
        <h2 className="mt-10 font-display text-[20px] font-bold tracking-tight text-ink">Single model</h2>
        <Endpoint method="GET" path="/api/v1/models/{id}">
          One model with full benchmark detail and the per-provider breakdown (pricing, quant,
          context and uptime by provider). Model ids contain a slash — pass them as-is.
        </Endpoint>
        <div className="mt-3">
          <Code>{`curl "${SITE}/api/v1/models/anthropic/claude-sonnet-4.5"`}</Code>
        </div>

        {/* Rankings */}
        <h2 className="mt-10 font-display text-[20px] font-bold tracking-tight text-ink">
          Rankings — best LLM for X
        </h2>
        <Endpoint method="GET" path="/api/v1/rankings/{collection}">
          The same ranked, answer-first lists that power the{" "}
          <Link href="/best" className="text-ink underline underline-offset-2">
            /best
          </Link>{" "}
          pages — each response includes a one-sentence <code className="font-mono">answer</code>.
          Optionally scope to a maker:{" "}
          <code className="font-mono">/api/v1/rankings/{"{collection}"}/{"{maker}"}</code>.
        </Endpoint>
        <div className="mt-3">
          <Code>{`# Best Anthropic model for coding
curl "${SITE}/api/v1/rankings/coding/anthropic"`}</Code>
        </div>
        <p className="mt-4 text-[13px] text-ink-2">
          <span className="font-semibold text-ink">Collections:</span>{" "}
          <code className="font-mono text-[12.5px]">small</code>
          {COLLECTIONS.map((c) => (
            <span key={c.slug}>
              {" · "}
              <code className="font-mono text-[12.5px]">{c.slug}</code>
            </span>
          ))}
        </p>

        {/* Makers */}
        <h2 className="mt-10 font-display text-[20px] font-bold tracking-tight text-ink">Makers</h2>
        <Endpoint method="GET" path="/api/v1/makers">
          Every model maker with model counts and its smartest, cheapest and fastest model — handy
          for building a maker filter against <code className="font-mono">/models</code>.
        </Endpoint>

        {/* FAQ — visible + FAQPage schema above */}
        <section className="mt-12">
          <h2 className="font-display text-[22px] font-bold tracking-tight text-ink">
            Frequently asked questions
          </h2>
          <div className="card-shadow mt-3 divide-y divide-line rounded-lg border border-line bg-surface">
            {FAQS.map((f) => (
              <div key={f.q} className="px-4 py-3.5">
                <h3 className="text-[15px] font-semibold text-ink">{f.q}</h3>
                <p className="mt-1.5 max-w-3xl text-sm leading-relaxed text-ink-2">{f.a}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Notes */}
        <h2 className="mt-10 font-display text-[20px] font-bold tracking-tight text-ink">Notes</h2>
        <ul className="mt-3 list-disc space-y-1.5 pl-5 text-[14px] leading-relaxed text-ink-2">
          <li>Prices are USD per million tokens. Speed is p50 output tokens/sec; latency is p50 time-to-first-token in ms.</li>
          <li>
            Benchmark fields are <code className="font-mono">null</code> when a model has no score —
            not every model is benchmarked.
          </li>
          <li>Data is sourced live from OpenRouter, Artificial Analysis and Design Arena, and refreshed continuously.</li>
          <li>This is an unofficial, independent project — not affiliated with OpenRouter.</li>
        </ul>

        {/* Internal link mesh */}
        <section className="mt-10">
          <h2 className="mb-2.5 text-[11px] font-semibold uppercase tracking-wider text-ink-3">Explore the data</h2>
          <div className="flex flex-wrap gap-2">
            {[
              { href: "/", label: "Leaderboard" },
              { href: "/best", label: "All rankings" },
              { href: "/compare", label: "Compare models" },
              { href: "/best/cheapest", label: "Cheapest LLMs" },
              { href: "/best/coding", label: "Best for coding" },
              { href: "/best/design", label: "Best for design" },
              { href: "/glossary", label: "Glossary" },
            ].map((l) => (
              <Link
                key={l.href}
                href={l.href}
                className="rounded-md border border-line bg-surface px-3 py-1.5 text-[12px] text-ink-2 hover:text-brand-ink"
              >
                {l.label}
              </Link>
            ))}
          </div>
        </section>
      </div>
      <Footer />
    </div>
  );
}
