/**
 * A derived "where you left off" card for long chats.
 *
 * Deliberately not `MemoryService.summarize()`: that produces a prompt fragment
 * ("Previous context: user: …") which is useful to a model and noise to a human.
 * This reads the messages the transcript already has, so it needs no model call,
 * no cache, and has nothing to go stale.
 */

export interface Digest {
  count: number;
  started: string;
  asking: string;
  answered: string;
}

export const DIGEST_MIN_MESSAGES = 12;

function flat(value: unknown, max: number): string {
  const text = String(value ?? "").replace(/\s+/g, " ").trim();
  if (!text) return "";
  return text.length > max ? `${text.slice(0, max - 1).trimEnd()}…` : text;
}

export function buildDigest(
  messages: { role: string; content: string }[],
): Digest | null {
  if (!Array.isArray(messages) || messages.length < DIGEST_MIN_MESSAGES) return null;

  const users = messages.filter(
    (message) => message.role === "user" && String(message.content ?? "").trim(),
  );
  const assistants = messages.filter(
    (message) =>
      message.role === "assistant" && String(message.content ?? "").trim(),
  );
  if (!users.length && !assistants.length) return null;

  return {
    count: messages.length,
    started: flat(users[0]?.content, 120),
    asking: flat(users[users.length - 1]?.content, 120),
    answered: flat(assistants[assistants.length - 1]?.content, 160),
  };
}
