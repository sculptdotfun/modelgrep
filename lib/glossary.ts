// Glossary of LLM terms — concise, authored definitions for the
// informational query space ("what is a context window", "what is GPQA").
// Each entry links back into the product so definitions convert to usage.

export interface GlossaryTerm {
  slug: string;
  term: string;
  short: string; // one-line definition (meta description / index)
  body: string[]; // paragraphs
  related: { label: string; href: string }[];
}

export const GLOSSARY: GlossaryTerm[] = [
  {
    slug: "context-window",
    term: "Context window",
    short:
      "The maximum amount of text (measured in tokens) an LLM can consider at once — its working memory for a single request.",
    body: [
      "A model's context window is the total number of tokens it can process in one request: your prompt, any documents you include, the conversation history, and the model's own output all share this budget. A 128K-token window fits roughly 300 pages of text; a 1M-token window fits a small codebase.",
      "Bigger isn't automatically better. Models often reason less reliably over information buried deep in a long context (the \"lost in the middle\" effect), long prompts cost more (you pay per input token), and latency grows with input size. The practical question is whether the model can actually use its full window, not just accept it.",
      "When choosing a model, match the window to the job: simple chat needs 8–32K, document analysis 100K+, and whole-repo coding or multi-document research benefits from 200K to 1M.",
    ],
    related: [
      { label: "Longest-context LLMs", href: "/best/long-context" },
      { label: "LLM leaderboard", href: "/" },
    ],
  },
  {
    slug: "tokens-per-second",
    term: "Tokens per second (throughput)",
    short:
      "The rate at which an LLM generates output, measured in tokens per second — the main determinant of how fast responses feel once they start.",
    body: [
      "Throughput measures how quickly a model produces output once it starts generating. At 30 tokens/sec a long answer streams in slowly; at 200+ tokens/sec it appears nearly instantly. A token is roughly ¾ of an English word.",
      "Throughput depends on both the model (smaller and quantized models are faster) and the inference provider's hardware. The same open-weight model can run at 40 t/s on one provider and 400 t/s on a provider using specialized accelerators.",
      "Throughput matters most for long outputs — code generation, long-form writing, agent loops that read their own output. For short answers, latency (time to first token) dominates perceived speed instead.",
    ],
    related: [
      { label: "Fastest LLMs", href: "/best/fastest" },
      { label: "Time to first token", href: "/glossary/time-to-first-token" },
    ],
  },
  {
    slug: "time-to-first-token",
    term: "Time to first token (latency)",
    short:
      "How long an LLM takes to begin responding after receiving a request — the metric that determines how responsive a model feels.",
    body: [
      "Time to first token (TTFT) is the delay between sending a prompt and receiving the first piece of the response. It covers network time, queueing, and \"prefill\" — the model processing your entire input before it can generate anything.",
      "Users perceive responses under ~300ms as instant and over ~1s as sluggish, so TTFT is the key metric for chat interfaces, autocomplete and any interactive product. Long prompts increase TTFT because prefill scales with input length; reasoning models can add seconds of hidden thinking before the first visible token.",
      "TTFT and throughput trade off differently by use case: interactive products should optimize TTFT first, batch pipelines can ignore it entirely.",
    ],
    related: [
      { label: "Lowest-latency LLMs", href: "/best/lowest-latency" },
      { label: "Tokens per second", href: "/glossary/tokens-per-second" },
    ],
  },
  {
    slug: "intelligence-index",
    term: "Artificial Analysis Intelligence Index",
    short:
      "A composite benchmark score (0–100) from Artificial Analysis that combines reasoning, knowledge, math and science evals into one comparable intelligence number.",
    body: [
      "The Intelligence Index is an independent composite score published by Artificial Analysis. It blends multiple hard evaluations — including GPQA Diamond (graduate-level science), Humanity's Last Exam, instruction following, and math/coding tasks — into a single number that tracks general capability.",
      "Because every model is run through the same harness, the index is one of the most reliable ways to compare models across labs — more robust than any single benchmark, and harder to game than self-reported scores.",
      "As a rough guide on today's scale: 50+ is frontier-class, 35–50 is strong production quality, 20–35 is capable for routine tasks, and below 20 suits narrow or high-volume budget work.",
    ],
    related: [
      { label: "Smartest LLMs", href: "/best/smartest" },
      { label: "GPQA", href: "/glossary/gpqa" },
    ],
  },
  {
    slug: "gpqa",
    term: "GPQA (Diamond)",
    short:
      "A graduate-level science benchmark of questions so hard that even skilled non-experts with web access score barely above chance — a standard test of deep reasoning.",
    body: [
      "GPQA (\"Graduate-Level Google-Proof Q&A\") is a multiple-choice benchmark written by PhD-level domain experts in biology, physics and chemistry. The questions are deliberately \"Google-proof\": non-experts with unlimited web access average only ~34%, barely above the 25% random baseline.",
      "GPQA Diamond is the hardest, highest-quality subset and the variant usually reported. Frontier models now score 70–90%+, making it one of the clearest separators between frontier and mid-tier models.",
      "Treat GPQA as a measure of deep scientific reasoning specifically — a model can score modestly on GPQA yet still be excellent at coding, writing, or agentic work.",
    ],
    related: [
      { label: "Smartest LLMs", href: "/best/smartest" },
      { label: "Intelligence Index", href: "/glossary/intelligence-index" },
    ],
  },
  {
    slug: "elo-rating",
    term: "Elo rating (for LLMs)",
    short:
      "A ranking system from competitive chess applied to AI: models battle head-to-head on the same task and humans pick the winner, producing a relative skill score.",
    body: [
      "An Elo rating expresses relative skill from pairwise battles. Two models receive the same prompt, humans (or judges) pick the better output, and the winner takes points from the loser — more points when an underdog wins. Over thousands of battles, ratings converge to a stable skill ordering.",
      "Elo-based leaderboards (like Design Arena for UI/frontend generation, or chatbot arenas for conversation) capture something static benchmarks miss: real human preference on open-ended work, where there is no single correct answer.",
      "Read Elo gaps probabilistically: a 100-point gap means the higher-rated model wins about 64% of battles; 200 points, about 76%. Small gaps (under ~30 points) are effectively ties.",
    ],
    related: [
      { label: "Best LLMs for design", href: "/best/design" },
      { label: "LLM leaderboard", href: "/" },
    ],
  },
  {
    slug: "prompt-caching",
    term: "Prompt caching",
    short:
      "A pricing and latency optimization where a provider reuses computation for repeated prompt prefixes, often cutting input costs by 50–90%.",
    body: [
      "When consecutive requests share a long common prefix — a system prompt, a document, a codebase — prompt caching lets the provider reuse the computation for that prefix instead of reprocessing it. Cached input tokens are billed at a steep discount (often 10–50% of the normal input price) and prefill latency drops too.",
      "Caching transforms the economics of agents and RAG systems, which resend large, mostly-identical contexts on every step. An agent loop that costs $1.00 per run without caching can cost $0.15 with it.",
      "Implementations differ: some providers cache implicitly and automatically, others require explicit cache-control markers, and cache lifetimes range from minutes to hours. Check the per-provider details on any model page.",
    ],
    related: [
      { label: "Cheapest LLMs", href: "/best/cheapest" },
      { label: "LLM pricing guide", href: "/blog/llm-pricing-explained" },
    ],
  },
  {
    slug: "quantization",
    term: "Quantization",
    short:
      "Compressing a model's weights to lower numeric precision (e.g. FP8, INT4) so it runs faster and cheaper — usually with a small quality cost.",
    body: [
      "Models are trained with 16-bit weights (BF16/FP16). Quantization stores those weights at lower precision — FP8 halves memory; INT4 quarters it — letting providers serve the same model on less hardware, faster and cheaper.",
      "The catch is quality: FP8 is usually near-lossless, while aggressive 4-bit quantization can measurably hurt reasoning and code generation. Two providers serving \"the same\" open-weight model may behave differently because one runs BF16 and the other INT4.",
      "When a provider lists a quantization tag (bf16, fp8, int4), factor it into price comparisons — the cheapest endpoint is sometimes cheap because it's a more heavily quantized serve.",
    ],
    related: [
      { label: "Open-source LLMs", href: "/best/open-source" },
      { label: "LLM leaderboard", href: "/" },
    ],
  },
  {
    slug: "reasoning-models",
    term: "Reasoning models",
    short:
      "LLMs that 'think' before answering — spending extra tokens on hidden chain-of-thought to solve harder problems at the cost of latency and price.",
    body: [
      "Reasoning models generate internal chains of thought before producing a final answer. This extra \"thinking\" dramatically improves performance on math, science, debugging and multi-step planning — the same base capability scores much higher when allowed to reason.",
      "The trade-offs are real: thinking tokens are billed as output (so costs rise), time-to-first-visible-token grows from milliseconds to seconds, and for simple tasks the extra deliberation adds nothing. Many models now expose a reasoning-effort dial so you can tune the trade-off per request.",
      "Use reasoning models for hard, high-value problems; use fast standard models for everyday completion, extraction and chat.",
    ],
    related: [
      { label: "Best reasoning LLMs", href: "/best/reasoning" },
      { label: "Best LLMs for agents", href: "/best/agents" },
    ],
  },
  {
    slug: "open-weights",
    term: "Open weights",
    short:
      "Models whose trained parameters are publicly downloadable — self-hostable, fine-tunable, and servable by any inference provider, unlike closed API-only models.",
    body: [
      "An open-weight model publishes its trained parameters (usually on Hugging Face), so anyone can download, run, fine-tune or re-serve it. Closed models (GPT, Claude, Gemini) are only reachable through their maker's API.",
      "Open weights bring price competition — dozens of providers serve the same model, driving costs down — plus data control (run it in your own VPC), customization via fine-tuning, and immunity to deprecation. The trade-off has historically been capability, but top open models now sit within a few points of frontier closed models on intelligence benchmarks.",
      "\"Open weights\" isn't the same as \"open source\": many licenses restrict commercial use or require attribution. Check the specific license before building on one.",
    ],
    related: [
      { label: "Best open-source LLMs", href: "/best/open-source" },
      { label: "Smartest LLMs", href: "/best/smartest" },
    ],
  },
];

export function getTerm(slug: string): GlossaryTerm | undefined {
  return GLOSSARY.find((t) => t.slug === slug);
}
