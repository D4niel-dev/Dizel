import type { Message, ToolCall } from "../types";

/** One thing the conversation produced: a file it wrote, or a block it wrote. */
export interface FileArtifact {
  key: string;
  tool: ToolCall;
  path: string;
  createdAt: number;
}

export interface CodeArtifact {
  key: string;
  language: string;
  code: string;
  createdAt: number;
}

const FENCE = /```([\w+#.-]*)\n([\s\S]*?)```/g;

/** Cache keyed by message id: a finished turn's code blocks never change. */
const codeCache = new Map<string, CodeArtifact[]>();

function blocksOf(message: Message): CodeArtifact[] {
  const cached = codeCache.get(message.id);
  if (cached) return cached;

  const found: CodeArtifact[] = [];
  for (const match of message.content.matchAll(FENCE)) {
    const code = match[2].replace(/\s+$/, "");
    // A two-line block is usually an example, not something to keep. The rail
    // is for artifacts; a stray `true` would only add noise.
    if (code.split("\n").length < 2) continue;
    found.push({
      key: `${message.id}:${found.length}`,
      language: (match[1] || "text").toLowerCase(),
      code,
      createdAt: message.createdAt,
    });
  }
  codeCache.set(message.id, found);
  return found;
}

/**
 * Everything the transcript produced, newest first.
 *
 * Tool calls are the reliable half: they carry a path, a diff or the content
 * that was written. Code blocks are taken only from finished turns — a block
 * still streaming would change identity on every token and lose the reader's
 * place in the list.
 */
export function collectArtifacts(messages: Message[]): {
  files: FileArtifact[];
  code: CodeArtifact[];
} {
  const files: FileArtifact[] = [];
  const code: CodeArtifact[] = [];

  for (const message of messages) {
    for (const tool of message.tools ?? []) {
      if (!tool.path && tool.diff === undefined && tool.content === undefined) continue;
      files.push({
        key: `${message.id}:${tool.id}`,
        tool,
        path: tool.path || tool.name,
        createdAt: message.createdAt,
      });
    }
    if (message.role === "assistant" && !message.streaming) {
      code.push(...blocksOf(message));
    }
  }

  return {
    files: files.reverse(),
    code: code.reverse(),
  };
}

/** The verb a row shows, derived from the tool rather than a second field. */
export function artifactVerb(tool: ToolCall): string {
  if (tool.name === "write_file") return tool.action === "create" ? "created" : "wrote";
  if (tool.name === "read_file") return "read";
  if (tool.diff !== undefined || tool.old !== undefined) return "edited";
  if (tool.content !== undefined) return "wrote";
  return tool.name.replace(/_/g, " ");
}

export function fileName(path: string): string {
  return path.split(/[\\/]/).pop() || path;
}

export function relativeTime(at: number, now = Date.now()): string {
  const seconds = Math.max(0, Math.round((now - at) / 1000));
  if (seconds < 60) return "just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}
