import { memo, useEffect, useId, useRef, useState } from "react";
import type { ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";

import type { Message, Prefs } from "../types";
import { Icon } from "./Icon";
import { ThinkingDots } from "./ThinkingDots";
import { formatBytes } from "./AttachmentPreview";
import { MessageFeedbackControls } from "./MessageFeedbackControls";
import { modelName } from "../services/model";
import { useChat } from "../store/chatStore";

function CodeBlock({ children }: { children?: ReactNode }) {
  const preRef = useRef<HTMLPreElement>(null);
  const [copied, setCopied] = useState(false);

  const first = Array.isArray(children) ? children[0] : children;
  const className =
    (first as { props?: { className?: string } } | undefined)?.props?.className ?? "";
  const language = /language-([\w+#.-]+)/.exec(className)?.[1] ?? "code";

  const copy = async () => {
    const text = preRef.current?.textContent ?? "";
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      /* clipboard unavailable */
    }
  };

  return (
    <div className="code-block">
      <div className="code-head">
        <span>{language}</span>
        <span className="spacer" />
        <button className="code-copy" onClick={() => void copy()}>
          <Icon name={copied ? "check" : "copy"} size={11} />
          {copied ? "copied" : "copy"}
        </button>
      </div>
      <pre ref={preRef}>{children}</pre>
    </div>
  );
}

function ExternalLink({ href, children }: { href?: string; children?: ReactNode }) {
  return (
    <a
      href={href}
      onClick={(event) => {
        event.preventDefault();
        if (href) void window.dizel?.openExternal(href);
      }}
    >
      {children}
    </a>
  );
}

const Markdown = memo(function Markdown({ text, highlight }: { text: string; highlight: boolean }) {
  return (
    <div className="md">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={highlight ? [[rehypeHighlight, { detect: true, ignoreMissing: true }]] : []}
        components={{ pre: CodeBlock, a: ExternalLink }}
      >
        {text.trimStart()}
      </ReactMarkdown>
    </div>
  );
});

const TYPING_FACTORS: Record<Prefs["typingAnimation"], number> = {
  off: 1,
  fast: 0.3,
  smooth: 0.14,
  cinematic: 0.07,
};

/**
 * Ease the rendered length toward the latest streamed chunk instead of snapping
 * on every token. The factor is the share of the backlog revealed per ~24 ms,
 * so bursts catch up fast while the reader never trails far behind the model.
 */
function useSmoothText(target: string, live: boolean, factor: number): string {
  // Existing text at mount (session switch mid-stream) is shown at once; only a
  // message that started empty replays from zero as chunks arrive.
  const [count, setCount] = useState(() =>
    live && factor < 1 && target.length === 0 ? 0 : target.length,
  );
  const countRef = useRef(count);
  const liveRef = useRef(live);
  const rafRef = useRef<number | null>(null);
  const lastRef = useRef(0);

  if (live) liveRef.current = true;

  useEffect(() => {
    if (!liveRef.current || factor >= 1) {
      countRef.current = target.length;
      setCount(target.length);
      return;
    }
    const tick = (now: number) => {
      if (now - lastRef.current < 32) {
        rafRef.current = window.requestAnimationFrame(tick);
        return;
      }
      lastRef.current = now;
      const remaining = target.length - countRef.current;
      if (remaining <= 0) {
        rafRef.current = null;
        return;
      }
      countRef.current = Math.min(
        target.length,
        countRef.current + Math.max(1, Math.ceil(remaining * factor)),
      );
      setCount(countRef.current);
      rafRef.current = window.requestAnimationFrame(tick);
    };
    if (rafRef.current === null) {
      rafRef.current = window.requestAnimationFrame(tick);
    }
    return () => {
      if (rafRef.current !== null) {
        window.cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [target, factor]);

  if (count >= target.length) return target;
  let end = count;
  // Never reveal half of a surrogate pair (emoji) for one frame.
  if (end > 0 && /[\uD800-\uDBFF]/.test(target[end - 1])) end -= 1;
  return target.slice(0, end);
}

/**
 * Attachments are folded into the stored prompt as <file> blocks. Reopening a
 * session would otherwise show the whole file dump in the user's bubble.
 */
function stripFileBlocks(text: string): string {
  const cleaned = text.replace(/<file name="[^"]*">[\s\S]*?<\/file>\s*/g, "").trim();
  return cleaned || text;
}

/**
 * Reasoning models wrap their scratchpad in <think> / <thinking>. Closed blocks
 * are stripped from the answer; a still-open block is shown as "Thinking…" so a
 * mid-stream message is never blank.
 */
function splitThinking(raw: string): { main: string; thoughts: string; pending: string } {
  const complete = /<(?:think|thinking)>([\s\S]*?)<\/(?:think|thinking)>/gi;
  const thoughts: string[] = [];

  for (const block of raw.match(complete) ?? []) {
    const inner = block
      .replace(/^<(?:think|thinking)>/i, "")
      .replace(/<\/(?:think|thinking)>$/i, "")
      .trim();
    if (inner) thoughts.push(inner);
  }

  let rest = raw.replace(complete, "");
  let pending = "";

  const openAt = rest.search(/<(?:think|thinking)>/i);
  if (openAt >= 0) {
    pending = rest
      .slice(openAt)
      .replace(/^<(?:think|thinking)>/i, "")
      .trim();
    rest = rest.slice(0, openAt);
  }

  return { main: rest.trim(), thoughts: thoughts.join("\n\n"), pending };
}

/** Reasoning is live; the line under the chip moves so a long pass never looks
 *  frozen. Kept short so the clock beside it fits without truncation; if a
 *  narrow window still runs out, the label wraps rather than clips.
 *  Presentation only — Dizel 1.5 has no reasoning pass. */
const THOUGHTS = [
  "Reading the question…",
  "Gathering context…",
  "Sketching an outline…",
  "Weighing approaches…",
  "Drafting a first pass…",
  "Tightening the wording…",
  "Still working through it…",
  "Checking assumptions…",
  "Comparing trade-offs…",
  "Testing the edge cases…",
  "Building the answer…",
  "Rereading the details…",
  "Working the hard part…",
  "Taking the time it needs…",
  "Finding the clearest path…",
  "Sorting signal from noise…",
  "Double-checking the logic…",
  "Choosing the right depth…",
  "Turning it into a plan…",
  "Keeping it grounded…",
];

function formatThoughtClock(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  if (total < 60) return `${total}s`;
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  if (hours > 0) return `${hours}h ${minutes}m`;
  return `${minutes}m ${total % 60}s`;
}

function thoughtLabel(pending: boolean, thoughtMs?: number): string {
  if (thoughtMs === undefined) return pending ? "Thinking" : "Thought";
  return `${pending ? "Thinking" : "Thought"} for ${formatThoughtClock(thoughtMs)}`;
}

function ThoughtBlock({
  text,
  pending,
  thoughtMs,
  thoughtStartedAt,
  highlight,
}: {
  text: string;
  pending: boolean;
  thoughtMs?: number;
  thoughtStartedAt?: number;
  highlight: boolean;
}) {
  // Open while the model is still reasoning; once the answer starts the chip
  // collapses and keeps only the "Thought for…" summary.
  const [open, setOpen] = useState(pending);
  const panelId = useId();
  const [mountedAt] = useState(() => Date.now());
  const [now, setNow] = useState(() => Date.now());
  const [wasPending, setWasPending] = useState(pending);
  // A long reasoning pass should read as alive. The phrase rotates on a slow
  // clock and the elapsed time stays beside it as the honest measure.
  const [phrase, setPhrase] = useState(() => Math.floor(Math.random() * THOUGHTS.length));
  // The reasoning pane follows the newest text while it streams, unless the
  // reader has scrolled up to read something earlier.
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const followRef = useRef(true);

  useEffect(() => {
    if (pending) {
      setOpen(true);
      setWasPending(true);
      followRef.current = true;
    }
  }, [pending]);

  useEffect(() => {
    if (!pending) return;
    const body = bodyRef.current;
    if (!body || !followRef.current) return;
    body.scrollTop = body.scrollHeight;
  }, [text, pending]);

  const onBodyScroll = () => {
    const body = bodyRef.current;
    if (!body) return;
    followRef.current = body.scrollHeight - body.scrollTop - body.clientHeight < 24;
  };

  useEffect(() => {
    if (!pending) return;
    const id = window.setInterval(
      () => setPhrase((index) => (index + 1) % THOUGHTS.length),
      3200,
    );
    return () => window.clearInterval(id);
  }, [pending]);

  const startedAt = thoughtStartedAt ?? mountedAt;
  const elapsed = Math.max(0, now - startedAt);
  // Inline <think> tags never produce a store-level start timestamp; keep the
  // locally observed clock as the final duration for those replies.
  const finalThoughtMs = thoughtMs ?? (wasPending ? elapsed : undefined);

  useEffect(() => {
    if (!pending) return;
    setNow(Date.now());
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [pending, startedAt]);

  return (
    <div className={`think${pending ? " is-pending" : ""}${open ? " is-open" : ""}`}>
      <button
        type="button"
        className="think-head"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        aria-controls={panelId}
        aria-label={
          pending
            ? `Thinking, ${formatThoughtClock(elapsed)}`
            : thoughtLabel(false, finalThoughtMs)
        }
      >
        <Icon name="sparkles" size={14} />
        <span className="think-summary">
          {pending ? (
            <>
              <span className="think-label" key={phrase}>
                {THOUGHTS[phrase]}
              </span>
              <span className="think-clock">{formatThoughtClock(elapsed)}</span>
            </>
          ) : (
            <span className="think-label">{thoughtLabel(false, finalThoughtMs)}</span>
          )}
        </span>
        <Icon name="chevronDown" size={12} />
      </button>
      <div className="think-panel" id={panelId} aria-hidden={!open}>
        <div className="think-panel-inner">
          <div className="think-body" ref={bodyRef} onScroll={onBodyScroll}>
            <Markdown text={text} highlight={highlight} />
          </div>
        </div>
      </div>
    </div>
  );
}

function metaLine(message: Message, prefs: Prefs, model: string): string {
  const parts: string[] = [];

  if (prefs.showModelTag && model) parts.push(model);

  if (message.durationMs && message.durationMs > 200) {
    parts.push(`${(message.durationMs / 1000).toFixed(1)}s`);
  }
  parts.push(`${message.content.length} chars`);

  // Throughput is the number that actually says how a local model is doing.
  // Estimated at 4 chars/token — the same ratio the context meter uses.
  if (prefs.showSpeed && message.durationMs && message.durationMs > 500) {
    const perSecond = Math.round(message.content.length / 4 / (message.durationMs / 1000));
    if (perSecond > 0) parts.push(`~${perSecond} tok/s`);
  }

  if (prefs.showTimestamps) {
    parts.push(
      new Date(message.createdAt).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        hour12: !prefs.clock24h,
      }),
    );
  }

  return parts.join(" · ");
}

export function MessageBubble({
  message,
  flash = false,
}: {
  message: Message;
  /** Momentarily highlighted after a find-in-conversation jump. */
  flash?: boolean;
}) {
  const regenerate = useChat((s) => s.regenerate);
  const streaming = useChat((s) => s.streaming);
  const prefs = useChat((s) => s.prefs);
  const status = useChat((s) => s.status);
  const setPreview = useChat((s) => s.setPreview);
  const beginEdit = useChat((s) => s.beginEdit);
  const showSibling = useChat((s) => s.showSibling);
  const showFork = useChat((s) => s.showFork);
  const continueLast = useChat((s) => s.continueLast);
  const [copied, setCopied] = useState(false);
  const holdMs = message.holdMs ?? 0;
  const [revealed, setRevealed] = useState(
    () => Date.now() - message.createdAt >= holdMs,
  );
  const [skipped, setSkipped] = useState(false);
  const [thought, setThought] = useState(0);
  const holding = !revealed && !skipped && holdMs > 0;

  // A reasoning channel or inline  thinking tags mean the ThoughtBlock owns the
  // progress UI. Everything else keeps the older rotating-line indicator.
  const hasReasoningSignal =
    Boolean(message.thinking) || /<(?:think|thinking)>/i.test(message.content);

  // Hold the answer for the full window even when the model finishes early —
  // a 1.1s reply would otherwise flash past the whole indicator.
  useEffect(() => {
    if (message.role !== "assistant") return;
    if (message.failed) {
      setRevealed(true);
      return;
    }
    const remaining = holdMs - (Date.now() - message.createdAt);
    if (remaining <= 0) {
      setRevealed(true);
      return;
    }
    setRevealed(false);
    const timer = window.setTimeout(() => setRevealed(true), remaining);
    return () => window.clearTimeout(timer);
  }, [message.createdAt, message.role, message.failed, holdMs]);

  useEffect(() => {
    if (message.role !== "assistant" || hasReasoningSignal) return;
    // Keep the line moving through the whole hold window, not just while the
    // model is streaming — a custom pause can outlive the reply by an hour.
    if (!holding && !message.streaming) return;
    const id = window.setInterval(() => setThought((value) => value + 1), 2400);
    return () => window.clearInterval(id);
  }, [hasReasoningSignal, holding, message.role, message.streaming]);

  const siblings = message.siblings;
  const view =
    siblings && siblings.length > 0
      ? siblings[Math.min(message.siblingIndex ?? 0, siblings.length - 1)]
      : message;
  const parsed = splitThinking(view.content);
  const typedMain = useSmoothText(
    parsed.main,
    message.role === "assistant" && Boolean(message.streaming),
    TYPING_FACTORS[prefs.typingAnimation],
  );

  // A handoff seeds the new chat with a system note so the model keeps context.
  // Rendering it as a bubble would make it look like something the user said.
  if (message.role === "system") {
    return (
      <div className="msg-system" data-mid={message.id}>
        <Icon name="archive" size={12} />
        <span>{message.content}</span>
      </div>
    );
  }

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(view.content);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      /* clipboard unavailable */
    }
  };

  if (message.role === "user") {
    return (
      <div className={flash ? "msg user is-found" : "msg user"} data-mid={message.id}>
        {/* Outside the bubble and above it — the files are what the prompt came
            with, not part of what was said. */}
        {message.attachments?.length ? (
          <div className="msg-attachments">
            {message.attachments.map((file) => (
              <button
                key={file.id}
                className="att-chip"
                onClick={() => setPreview(file)}
                title={`${file.name} — click to read`}
              >
                <Icon name="file" size={12} />
                <span className="att-name">{file.name}</span>
                <span className="att-size">{formatBytes(file.size)}</span>
              </button>
            ))}
          </div>
        ) : null}

        <div className="user-card">
          <div className="msg-label">You</div>
          <div className="user-text">{stripFileBlocks(message.content)}</div>
        </div>

        {!streaming && (
          <div className="msg-actions user-actions">
            <button className="act" onClick={() => beginEdit(message.id)} title="Edit and resend">
              <Icon name="compose" size={13} />
              Edit
            </button>

            {message.forks && message.forks.length > 1 && (
              <span className="sib-switch">
                <button
                  className="sib-btn"
                  onClick={() => showFork(message.id, -1)}
                  title="Previous version of this question"
                  aria-label="Previous version"
                >
                  <Icon name="chevronUp" size={12} style={{ transform: "rotate(-90deg)" }} />
                </button>
                <span className="sib-count">
                  {(message.forkIndex ?? 0) + 1}/{message.forks.length}
                </span>
                <button
                  className="sib-btn"
                  onClick={() => showFork(message.id, 1)}
                  title="Next version of this question"
                  aria-label="Next version"
                >
                  <Icon name="chevronDown" size={12} style={{ transform: "rotate(-90deg)" }} />
                </button>
              </span>
            )}
          </div>
        )}
      </div>
    );
  }

  // Two sources of reasoning: a provider that streams it on its own channel
  // (Ollama's `thinking`, Anthropic's `thinking_delta`), and local models that
  // emit inline  thinking…<｜end▁of▁thinking｜> tags. Both land in the same block.
  // When this reply has siblings, the bubble renders whichever one is selected.
  // Everything visible (content, reasoning, meta, copy) reads from `view`; the
  // live streaming state still reads from `message`.
  const streamed = view.thinking ?? "";
  const thoughtText = [streamed, parsed.thoughts, parsed.pending]
    .filter(Boolean)
    .join("\n\n");
  const isPending =
    Boolean(message.streaming) && (Boolean(parsed.pending) || (streamed.length > 0 && !parsed.main));

  // Reasoning gets the ThoughtBlock; everything else keeps the older rotating
  // line while it waits for the first answer token.
  const hasThought = Boolean(thoughtText);
  const waitingForText = Boolean(message.streaming) && !parsed.main && !hasThought;
  const showGenerating = !hasThought && (holding || waitingForText);

  // `max_new_tokens` cuts a reply wherever the budget runs out, which in practice
  // is mid-word. A finished sentence ends in punctuation; a truncated one does not.
  const trimmedContent = view.content.trimEnd();
  const looksCut =
    trimmedContent.length > 80 && /[A-Za-z0-9,;:\-–—]$/.test(trimmedContent);

  return (
    <div
      className={
        message.failed
          ? "msg assistant failed"
          : flash
            ? "msg assistant is-found"
            : "msg assistant"
      }
      data-mid={message.id}
    >
      {showGenerating &&
        (holding ? (
          <button
            type="button"
            className="gen-row"
            onClick={() => setSkipped(true)}
            title="Show the answer now"
            aria-label="Generating — click to show the answer"
          >
            <ThinkingDots />
            <span className="gen-thought" key={thought}>
              {THOUGHTS[thought % THOUGHTS.length]}
            </span>
          </button>
        ) : (
          <div className="gen-row is-static" aria-hidden="true">
            <ThinkingDots />
            <span className="gen-thought" key={thought}>
              {THOUGHTS[thought % THOUGHTS.length]}
            </span>
          </div>
        ))}

      {hasThought && (
        <ThoughtBlock
          text={thoughtText}
          pending={isPending}
          thoughtMs={view.thoughtMs}
          thoughtStartedAt={view.thoughtStartedAt}
          highlight={prefs.syntaxHighlight}
        />
      )}

      {!holding && parsed.main && (
        <Markdown text={typedMain} highlight={prefs.syntaxHighlight} />
      )}
      {message.streaming && !isPending && !holding && parsed.main && <span className="caret" />}

      {!message.streaming && !holding && (
        <div className="msg-actions">
          <button className="act" onClick={() => void copy()}>
            <Icon name={copied ? "check" : "copy"} size={13} />
            {copied ? "Copied" : "Copy"}
          </button>
          <button className="act" onClick={() => void regenerate()} disabled={streaming}>
            <Icon name="refresh" size={13} />
            Regenerate
          </button>

          <MessageFeedbackControls messageId={message.id} content={view.content} />

          {looksCut && (
            <button
              className="act"
              onClick={() => void continueLast(message.id)}
              disabled={streaming}
              title="Keep going from where it stopped"
            >
              <Icon name="chevronDown" size={13} />
              Continue
            </button>
          )}

          {siblings && siblings.length > 1 && (
            <span className="sib-switch">
              <button
                className="sib-btn"
                onClick={() => showSibling(message.id, -1)}
                title="Previous reply"
                aria-label="Previous reply"
              >
                <Icon name="chevronUp" size={12} style={{ transform: "rotate(-90deg)" }} />
              </button>
              <span className="sib-count">
                {(message.siblingIndex ?? 0) + 1}/{siblings.length}
              </span>
              <button
                className="sib-btn"
                onClick={() => showSibling(message.id, 1)}
                title="Next reply"
                aria-label="Next reply"
              >
                <Icon name="chevronDown" size={12} style={{ transform: "rotate(-90deg)" }} />
              </button>
            </span>
          )}

          <span className="spacer" />
          <span className="msg-meta">{metaLine(view, prefs, modelName(status))}</span>
        </div>
      )}
    </div>
  );
}
