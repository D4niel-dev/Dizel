import type { Message } from "../types";

/**
 * Reply variants, kept beside the session rather than inside it.
 *
 * The server transcript holds exactly one reply per turn, so the alternatives
 * live here keyed by the session id and the *assistant turn number*. Turn order
 * is stable across a reload — the server returns the same messages in the same
 * order — which makes it a safer key than a message id the client generates.
 */
const key = (sessionId: string) => `dizel.variants.${sessionId}`;

export type VariantMap = Record<string, Message[]>;

export function loadVariants(sessionId: string | null): VariantMap {
  if (!sessionId) return {};
  try {
    const raw = localStorage.getItem(key(sessionId));
    return raw ? (JSON.parse(raw) as VariantMap) : {};
  } catch {
    return {};
  }
}

export function saveVariants(sessionId: string | null, map: VariantMap): void {
  if (!sessionId) return;
  try {
    localStorage.setItem(key(sessionId), JSON.stringify(map));
  } catch {
    /* quota — variants are a convenience, never load-bearing */
  }
}

/** Which assistant turn a message is, counting from zero. -1 if it is not one. */
export function assistantTurnIndex(messages: Message[], messageId: string): number {
  let turn = -1;
  for (const message of messages) {
    if (message.role === "assistant") turn += 1;
    if (message.id === messageId) return message.role === "assistant" ? turn : -1;
  }
  return -1;
}

/** Re-attach saved variants to a freshly loaded transcript. */
export function applyVariants(messages: Message[], saved: VariantMap): Message[] {
  let turn = -1;
  return messages.map((message) => {
    if (message.role !== "assistant") return message;
    turn += 1;
    const siblings = saved[String(turn)];
    if (!siblings?.length) return message;
    return { ...message, siblings, siblingIndex: siblings.length - 1 };
  });
}
