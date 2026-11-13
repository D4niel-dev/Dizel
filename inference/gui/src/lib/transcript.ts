import type { Message } from "../types";

/** A filename that sorts and reads well: dizel-2026-11-05-1652.md */
export function exportName(prefix = "dizel"): string {
  const now = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  const stamp =
    `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}` +
    `-${pad(now.getHours())}${pad(now.getMinutes())}`;
  return `${prefix}-${stamp}.md`;
}

/**
 * The transcript as Markdown.
 *
 * Reasoning is deliberately left out — it is not part of what was said, and a
 * reader of an exported file has no way to tell it apart from the answer.
 * Attachments are kept as a line on the message that carried them.
 */
export function transcriptToMarkdown(messages: Message[], title: string): string {
  const lines: string[] = [];
  const now = new Date();

  lines.push(`# ${title || "Dizel conversation"}`);
  lines.push("");
  lines.push(`_Exported ${now.toLocaleString()}_`);
  lines.push("");

  for (const message of messages) {
    if (message.role === "system") {
      lines.push(`> ${message.content.replace(/\n/g, "\n> ")}`);
      lines.push("");
      continue;
    }

    lines.push(`## ${message.role === "user" ? "You" : "Dizel"}`);
    lines.push("");

    if (message.attachments?.length) {
      const names = message.attachments.map((file) => file.name).join(", ");
      lines.push(`*Attached: ${names}*`);
      lines.push("");
    }

    const body = message.content.trim();
    lines.push(body || "_(empty)_");
    lines.push("");
  }

  return lines.join("\n").replace(/\n{3,}/g, "\n\n").trimEnd() + "\n";
}
