// Shared types for the modelgrep catalog.

export interface Capabilities {
  tools: boolean;
  reasoning: boolean;
  structured: boolean;
  vision: boolean;
  audio_in: boolean;
  image_out: boolean;
}

export interface AABenchmark {
  intelligence: number | null;
  coding: number | null;
  agentic: number | null;
  gpqa: number | null;
  hle: number | null;
  scicode: number | null;
  tau2: number | null;
  intelligence_pct: number | null;
  coding_pct: number | null;
  agentic_pct: number | null;
}

export interface DACategory {
  elo: number | null;
  win_rate: number | null;
}

export interface DABenchmark {
  elo: number | null;
  category: string | null;
  win_rate: number | null;
  elo_pct: number | null;
  tournaments: number | null;
  categories: Record<string, DACategory>;
}

export interface ProviderDetail {
  name: string;
  quantization: string;
  context_length: number;
  max_completion: number | null;
  price_input: number | null;
  price_output: number | null;
  uptime: number | null;
  caching: boolean;
}

export interface Model {
  id: string;
  canonical_slug: string;
  name: string;
  description: string;
  context_length: number;
  max_output: number | null;
  throughput: number; // tokens/sec, p50 (0 when unknown)
  latency: number | null; // ms, p50
  uptime: number | null; // best provider uptime %
  price_input: number | null; // $/1M tokens
  price_output: number | null;
  price_cache_read: number | null;
  supports_caching: boolean;
  providers: string[];
  modality: string;
  input_modalities: string[];
  output_modalities: string[];
  capabilities: Capabilities;
  knowledge_cutoff: string | null;
  is_moderated: boolean | null;
  hugging_face_id: string | null;
  created: number;
  aa: AABenchmark | null;
  da: DABenchmark | null;
}

// Per-model detail adds the live provider breakdown (fetched on the model page).
export interface ModelDetail extends Model {
  provider_details: ProviderDetail[];
}
