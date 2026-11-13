import type { ChatStatus } from "../types";

const PROVIDER_NAMES: Record<string, string> = {
  local: "Local",
  openai: "OpenAI",
  anthropic: "Anthropic",
  google: "Google",
  groq: "Groq",
  mistral: "Mistral AI",
  xai: "xAI",
  cohere: "Cohere",
  ollama: "Ollama",
  azure: "Azure OpenAI",
  ai21: "AI21 Labs",
  meta: "Meta (Llama)",
  custom: "Custom",
};

/** Display name for the active provider slug, with a safe prettifier fallback. */
export function providerName(status: ChatStatus | null): string {
  const slug = (status?.provider || "local").toLowerCase();
  const known = PROVIDER_NAMES[slug];
  if (known) return known;
  return slug
    .split(/[-_]/)
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

/** Human-readable model name from whatever the backend currently reports. */
export function modelName(status: ChatStatus | null): string {
  const displayName = status?.load?.display_name;
  if (displayName) return displayName;

  const checkpoint = status?.load?.checkpoint ?? "";
  if (checkpoint) {
    const file = checkpoint.split(/[\\/]/).pop() ?? "";
    return file.replace(/\.(gguf|pt|safetensors)$/i, "");
  }
  return status?.api_model || "Model";
}
