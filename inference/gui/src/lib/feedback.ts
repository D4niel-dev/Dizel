import type { MessageFeedback } from "../types";

/**
 * Feedback lives beside the session, keyed by assistant turn and a content
 * hash. The hash keeps a rating attached to the exact reply when variants are
 * switched or a regenerated answer becomes a sibling.
 */
const key = (sessionId: string) => `dizel.feedback.${sessionId}`;

export type FeedbackMap = Record<string, MessageFeedback>;

export function feedbackHash(text: string): string {
  let hash = 5381;
  for (let index = 0; index < text.length; index += 1) {
    hash = ((hash << 5) + hash + text.charCodeAt(index)) | 0;
  }
  return (hash >>> 0).toString(36);
}

export function feedbackKey(turn: number, content: string): string {
  return `${turn}:${feedbackHash(content)}`;
}

function loadMap(sessionId: string | null): FeedbackMap {
  if (!sessionId) return {};
  try {
    const raw = localStorage.getItem(key(sessionId));
    const parsed = raw ? (JSON.parse(raw) as unknown) : null;
    return parsed && typeof parsed === "object" ? (parsed as FeedbackMap) : {};
  } catch {
    return {};
  }
}

export function getFeedback(
  sessionId: string | null,
  turn: number,
  content: string,
): MessageFeedback | null {
  if (!sessionId || turn < 0) return null;
  const entry = loadMap(sessionId)[feedbackKey(turn, content)];
  if (!entry || (entry.rating !== "up" && entry.rating !== "down")) return null;
  return entry;
}

export function setFeedbackEntry(
  sessionId: string | null,
  turn: number,
  content: string,
  entry: MessageFeedback | null,
): void {
  if (!sessionId || turn < 0) return;
  const map = loadMap(sessionId);
  const mapKey = feedbackKey(turn, content);
  if (entry) map[mapKey] = entry;
  else delete map[mapKey];
  try {
    localStorage.setItem(key(sessionId), JSON.stringify(map));
  } catch {
    /* quota — feedback is a local convenience */
  }
}
