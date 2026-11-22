import { createPortal } from "react-dom";

import type { ContextUsage } from "../services/api";

/**
 * What the context window actually holds.
 *
 * The window is small and the system prompt is a fixed tax on it, so the useful
 * question is not "how full is it" but "what is in there, and what did the
 * trimmer drop". The status bar answers the first; this answers the second.
 */

interface Segment {
  key: string;
  label: string;
  tokens: number;
  color: string;
  child?: boolean;
}

const COLORS = {
  base: "#8b8d90",
  mode: "var(--accent)",
  project: "#4fd6be",
  verbosity: "#fbbf24",
  conversation: "#a78bfa",
  reserved: "#fb923c",
};

export function ContextPanel({
  usage,
  anchor,
}: {
  usage: ContextUsage | null;
  anchor: { left: number; top: number; width: number } | null;
}) {
  const total = usage?.total ?? 0;
  const used = usage?.used ?? 0;
  const reserved = usage?.reserved ?? 0;
  const trimmed = usage?.trimmed ?? 0;
  const free = Math.max(0, total - used);

  // An older sidecar does not send system_blocks; fall back to one system row
  // rather than showing an empty panel.
  const blocks = usage?.system_blocks;
  const segments: Segment[] = blocks
    ? [
        { key: "base", label: "System prompt", tokens: blocks.base, color: COLORS.base },
        {
          key: "mode",
          label: "Mode instructions",
          tokens: blocks.mode,
          color: COLORS.mode,
          child: true,
        },
        {
          key: "verbosity",
          label: "Response length",
          tokens: blocks.verbosity ?? 0,
          color: COLORS.verbosity,
          child: true,
        },
        {
          key: "project",
          label: "Project context",
          tokens: blocks.project,
          color: COLORS.project,
          child: true,
        },
        {
          key: "conversation",
          label: "Conversation",
          tokens: usage?.conversation ?? 0,
          color: COLORS.conversation,
        },
      ]
    : [
        {
          key: "system",
          label: "System prompt",
          tokens: usage?.system ?? 0,
          color: COLORS.base,
        },
        {
          key: "conversation",
          label: "Conversation",
          tokens: usage?.conversation ?? 0,
          color: COLORS.conversation,
        },
      ];

  if (reserved > 0) {
    segments.push({
      key: "reserved",
      label: "Reply reserve",
      tokens: reserved,
      color: COLORS.reserved,
    });
  }

  // The status bar clips its overflow, so the panel mounts on the body and is
  // placed above the trigger instead of being a child of it.
  const width = 266;
  // Positioned with , not : the popover-in animation sets a
  // transform, and an animation beats an inline style, so translateY was ignored
  // and the panel rendered downward off-screen.
  const left = anchor
    ? Math.max(8, Math.min(anchor.left, window.innerWidth - width - 12))
    : 0;
  // The caret points at the meter, not at the panel's centre: the panel is
  // clamped to the viewport, so the two are often not the same.
  const caret = anchor
    ? Math.max(14, Math.min(anchor.left + anchor.width / 2 - left, width - 14))
    : width / 2;
  const style = anchor
    ? {
        position: "fixed" as const,
        left,
        bottom: window.innerHeight - anchor.top + 10,
        ["--ctx-caret" as string]: `${caret}px`,
      }
    : { display: "none" as const };

  return createPortal(
    <div className="ctx-panel" style={style} role="dialog" aria-label="Context window">
      <div className="ctx-panel-head">
        <span>Context window</span>
        <b>
          {used.toLocaleString()} / {total.toLocaleString()}
        </b>
      </div>

      <div className="ctx-stack" aria-hidden="true">
        {segments.map((segment) => (
          <span
            key={segment.key}
            className="ctx-stack-seg"
            style={{
              flexGrow: Math.max(segment.tokens, 0),
              background: segment.color,
            }}
            title={`${segment.label} — ${segment.tokens}`}
          />
        ))}
        <span className="ctx-stack-free" style={{ flexGrow: Math.max(free, 0) }} />
      </div>

      <div className="ctx-rows">
        {segments.map((segment) => (
          <div
            key={segment.key}
            className={segment.child ? "ctx-row is-child" : "ctx-row"}
          >
            <i style={{ background: segment.color }} aria-hidden="true" />
            <span>{segment.label}</span>
            <b>{segment.tokens.toLocaleString()}</b>
          </div>
        ))}
      </div>

      <div className="ctx-panel-foot">
        <span>{free.toLocaleString()} free</span>
        {trimmed > 0 && <span className="ctx-panel-trim">{trimmed} trimmed</span>}
        {usage?.task && <span className="ctx-panel-task">{usage.task}</span>}
      </div>
    </div>,
    document.body,
  );
}
