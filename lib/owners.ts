// Per-owner brand accent colors for fast visual scanning in the leaderboard.
// Falls back to a deterministic hue derived from the owner slug.

const BRAND: Record<string, string> = {
  openai: "#10a37f",
  anthropic: "#d97757",
  google: "#4285f4",
  "meta-llama": "#0866ff",
  "x-ai": "#a1a1aa",
  mistralai: "#fa520f",
  deepseek: "#4d6bfe",
  qwen: "#615ced",
  "z-ai": "#615ced",
  moonshotai: "#7c5cff",
  cohere: "#39a0a0",
  microsoft: "#00a4ef",
  amazon: "#ff9900",
  nvidia: "#76b900",
  perplexity: "#20808d",
  "ai21": "#e8488a",
  inflection: "#ff6b6b",
  nousresearch: "#22c55e",
  inclusionai: "#6366f1",
  morph: "#a855f7",
  liquid: "#0ea5e9",
};

function hashHue(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) % 360;
  return h;
}

export function ownerColor(owner: string): string {
  const key = owner.toLowerCase();
  if (BRAND[key]) return BRAND[key];
  return `hsl(${hashHue(key)} 55% 60%)`;
}
