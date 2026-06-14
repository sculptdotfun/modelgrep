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
  {
    slug: "parameters",
    term: "Parameters (model size)",
    short:
      "The trained weights inside a model, counted in billions (B). More parameters generally means more capability — and more cost and latency to run.",
    body: [
      "A model's parameters are the numbers it learned during training — the knobs that encode everything it knows. Count is quoted in billions: an 8B model has 8 billion weights, a 70B model has 70 billion. It's the closest thing to a model's \"size.\"",
      "More parameters usually means more capability, but also more memory, higher price and slower inference. That's why labs ship tiers: a small 8B model for cheap high-volume work, a large flagship for the hardest tasks. The frontier closed models (GPT, Claude, Gemini) don't disclose parameter counts at all — so for them, \"small\" means the efficient tier (Haiku, mini, Flash), not a public number.",
      "Don't over-index on raw size. Architecture, training data and post-training matter just as much: a well-trained 30B model routinely beats an older 70B one. Judge by benchmarks and price-performance, not parameter count alone.",
    ],
    related: [
      { label: "Small & fast LLMs", href: "/best/small" },
      { label: "Mixture of experts", href: "/glossary/mixture-of-experts" },
    ],
  },
  {
    slug: "mixture-of-experts",
    term: "Mixture of experts (MoE)",
    short:
      "An architecture that splits a model into many specialized \"expert\" sub-networks and activates only a few per token — large total size, small active cost.",
    body: [
      "A mixture-of-experts model contains many expert sub-networks but routes each token to just a handful of them. A model can have 235B total parameters yet activate only ~22B per token — so it carries the knowledge of a huge model while costing closer to a small one to run.",
      "This is why modern open-weight leaders (DeepSeek, Qwen, Llama 4, Mixtral) are nearly all MoE: it decouples capability from inference cost. The tradeoff is memory — you still have to load all the experts — and routing complexity.",
      "When you see a model quoted as \"A active / B total\" (e.g. 37B active / 671B total), that's MoE. Price and speed track the active count; capability tracks the total.",
    ],
    related: [
      { label: "Parameters", href: "/glossary/parameters" },
      { label: "Best open-source LLMs", href: "/best/open-source" },
    ],
  },
  {
    slug: "function-calling",
    term: "Function calling (tool use)",
    short:
      "A model capability for emitting structured calls to external tools or APIs — the foundation of agents, retrieval and anything that acts on the world.",
    body: [
      "Function calling (a.k.a. tool use) lets a model output a structured request — a function name plus JSON arguments — instead of plain text. Your code runs the function, returns the result, and the model continues. It's how an LLM checks the weather, queries a database, or runs a calculation it can't do reliably itself.",
      "Reliable tool use is the single hardest requirement for agents, which chain dozens of calls and must pick the right tool with the right arguments every time. The Artificial Analysis Agentic Index and Tau²-Bench exist specifically to measure this.",
      "Not every model supports it, and support quality varies widely — a model can be brilliant at prose yet unreliable at emitting valid tool calls. Filter for the tool-calling capability when building anything agentic.",
    ],
    related: [
      { label: "Best LLMs for agents", href: "/best/agents" },
      { label: "Reasoning models", href: "/glossary/reasoning-models" },
    ],
  },
  {
    slug: "retrieval-augmented-generation",
    term: "Retrieval-augmented generation (RAG)",
    short:
      "A pattern that fetches relevant documents at query time and feeds them into the prompt, so the model answers from your data instead of memory.",
    body: [
      "RAG retrieves relevant chunks from a knowledge base — usually via vector search — and pastes them into the context before the model answers. It grounds responses in your own, current documents rather than the model's frozen training data, which cuts hallucination and sidesteps the knowledge cutoff.",
      "RAG systems resend large, mostly-identical context on every query, so two model traits dominate the bill and the latency: a context window big enough to hold retrieved passages, and prompt caching to avoid re-billing the shared prefix. The two together decide whether a RAG product is cheap or ruinous at scale.",
      "RAG and long context are complements, not rivals: retrieval narrows millions of tokens down to the few thousand that matter, and a capable context window reasons over them.",
    ],
    related: [
      { label: "Prompt caching", href: "/glossary/prompt-caching" },
      { label: "Longest-context LLMs", href: "/best/long-context" },
    ],
  },
  {
    slug: "knowledge-cutoff",
    term: "Knowledge cutoff",
    short:
      "The date after which a model has no built-in knowledge — anything more recent must be supplied via tools, search or your prompt.",
    body: [
      "A model's knowledge cutoff is the last date covered by its training data. Ask about an event after that date and a model will either say it doesn't know or, worse, confidently invent an answer. Cutoffs typically lag the release date by several months to a year.",
      "The cutoff only bounds built-in memory, not what a model can do. Give it web search, function calling or RAG and it works with live information regardless of when it was trained. For anything time-sensitive, design for retrieval rather than trusting recall.",
      "Each model page on modelgrep lists the knowledge cutoff where the maker discloses it, alongside the release date — the gap between them is a useful tell about how current a model's baked-in knowledge is.",
    ],
    related: [
      { label: "New model releases", href: "/new" },
      { label: "Retrieval-augmented generation", href: "/glossary/retrieval-augmented-generation" },
    ],
  },
  {
    slug: "model-tiers",
    term: "Model tiers (small, mid, frontier)",
    short:
      "Most labs ship a family at three rough tiers — a small fast one, a balanced one, and a flagship — so you can match capability to the job and the budget.",
    body: [
      "Nearly every lab ships a tiered family rather than one model. Anthropic has Haiku / Sonnet / Opus; OpenAI has nano / mini / full; Google has Flash / Pro. The pattern is the same: a small, fast, cheap tier for high volume; a flagship for the hardest reasoning; and a balanced middle that's the right default for most production work.",
      "The small tier is the workhorse most teams underuse. For chat, classification, extraction and routing, a Haiku- or Flash-class model is often 5–20× cheaper and several times faster than the flagship, at quality the task can't actually distinguish. Reach for the flagship only where reasoning genuinely gates the outcome.",
      "A common production pattern is to route by difficulty — cheap small model first, escalate to the flagship only when confidence is low — which captures most of the quality at a fraction of the cost.",
    ],
    related: [
      { label: "Small & fast LLMs", href: "/best/small" },
      { label: "LLM speed vs cost", href: "/blog/speed-vs-cost" },
    ],
  },
  {
    slug: "swe-bench",
    term: "SWE-bench",
    short:
      "A coding benchmark of real GitHub issues — the model must produce a patch that makes the repo's actual test suite pass. The closest thing to a real-world software eval.",
    body: [
      "SWE-bench draws from genuine resolved issues in popular open-source Python repos. The model gets the codebase and the issue text and must generate a patch; it's scored by whether the repository's own hidden tests then pass. That makes it far harder to game than a multiple-choice quiz — and a strong proxy for agentic coding ability.",
      "SWE-bench Verified is the human-validated subset most often reported, since the original set contained some unsolvable or ambiguous tasks. It's a core ingredient in the Artificial Analysis Coding Index.",
      "Treat it as a measure of autonomous bug-fixing in real repos specifically. A high SWE-bench score signals a model that can navigate a codebase and use tools, which correlates with — but isn't identical to — good interactive code completion.",
    ],
    related: [
      { label: "Best LLMs for coding", href: "/best/coding" },
      { label: "Best LLMs for agents", href: "/best/agents" },
    ],
  },
  {
    slug: "multimodal",
    term: "Multimodal model",
    short:
      "A model that handles more than text — most commonly accepting image input alongside text, and sometimes audio in or image/audio out.",
    body: [
      "A multimodal model accepts or produces more than one type of data. In practice this almost always means vision: the model reads images, screenshots, charts, PDFs and diagrams as naturally as text. Some also take audio input, and a few generate images or speech as output.",
      "Vision unlocks whole categories of work text-only models can't touch — document and receipt extraction, UI understanding, chart reading, visual QA, accessibility. It's a hard requirement for those, and irrelevant for pure text pipelines, so filter for it deliberately.",
      "\"Multimodal\" doesn't imply every modality. Check the specific input and output modalities on a model's page: many accept images but only emit text, and audio or image generation is still comparatively rare.",
    ],
    related: [
      { label: "Best vision LLMs", href: "/best/vision" },
      { label: "LLM leaderboard", href: "/" },
    ],
  },
  {
    slug: "fine-tuning",
    term: "Fine-tuning",
    short:
      "Continuing training on your own examples to specialize a base model's style, format or domain — distinct from just writing a better prompt.",
    body: [
      "Fine-tuning updates a model's weights on a curated dataset of your examples, baking in a behavior rather than describing it in the prompt every time. It's the tool for locking in a house style, a rigid output format, or a narrow domain vocabulary that prompting alone keeps drifting away from.",
      "It's usually the wrong first move. Prompting, few-shot examples and RAG solve most problems faster and cheaper, and a fine-tuned model is frozen — it won't benefit from the next base-model upgrade without redoing the work. Reach for it only after the cheaper levers plateau.",
      "Fine-tuning needs open weights or a provider's tuning API. Open-weight models (on Hugging Face) give you full control; closed models offer it only where the maker exposes it.",
    ],
    related: [
      { label: "Open weights", href: "/glossary/open-weights" },
      { label: "Best open-source LLMs", href: "/best/open-source" },
    ],
  },
  {
    slug: "temperature",
    term: "Temperature",
    short:
      "A sampling setting (typically 0–2) that controls randomness: low is deterministic and focused, high is varied and creative.",
    body: [
      "Temperature scales how sharply a model favors its most likely next token. Near 0 it picks the top choice almost every time — repeatable, focused, ideal for extraction, classification, code and anything with a single right answer. Higher values flatten the distribution, adding variety useful for brainstorming and creative writing.",
      "There's no universally \"correct\" value — it's per-task. A rule of thumb: 0–0.3 for structured or factual work, 0.7–1.0 for open-ended generation. If you need reproducible outputs (tests, evals, caching), pin it low.",
      "Temperature is independent of the model: the same setting behaves differently across models, and reasoning models often manage their own internal sampling, making the dial less impactful for them.",
    ],
    related: [
      { label: "Reasoning models", href: "/glossary/reasoning-models" },
      { label: "How to choose an LLM", href: "/blog/how-to-choose-llm" },
    ],
  },
];

export function getTerm(slug: string): GlossaryTerm | undefined {
  return GLOSSARY.find((t) => t.slug === slug);
}
