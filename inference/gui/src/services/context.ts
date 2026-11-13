import type { Message } from "../types";

export interface ContextUsage {
  used: number;
  total: number;
  percent: number;
}

/**
 * Rough context usage. The backend budgets in tokens but doesn't report a count
 * back to the renderer, so this estimates at ~4 characters per token — hence the
 * "~" wherever it's displayed.
 */
export function contextUsage(messages: Message[], ctxLen?: number | null): ContextUsage {
  const total = Number(ctxLen ?? 2048) || 2048;
  const characters = messages.reduce((sum, message) => sum + message.content.length, 0);
  const used = Math.round(characters / 4);

  return { used, total, percent: Math.min(100, Math.round((used / total) * 100)) };
}
