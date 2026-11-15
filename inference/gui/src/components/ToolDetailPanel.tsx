import { useMemo, useState } from "react";
import type { ReactNode } from "react";
import hljs from "highlight.js/lib/common";

import type { ToolCall } from "../types";
import { kindOf } from "./AttachmentPreview";
import { Icon } from "./Icon";

function openUrl(url: string): void {
  if (!url) return;
  if (window.dizel?.openExternal) {
    void window.dizel.openExternal(url);
  } else {
    window.open(url, "_blank", "noopener,noreferrer");
  }
}

type DiffKind = "meta" | "hunk" | "add" | "del" | "ctx";

interface DiffRow {
  kind: DiffKind;
  number: number | null;
  sign: string;
  text: string;
}

function parseDiff(diff: string): DiffRow[] {
  const rows: DiffRow[] = [];
  let oldLine = 0;
  let newLine = 0;
  let inHunk = false;

  for (const line of diff.split("\n")) {
    if (line.startsWith("---") || line.startsWith("+++") || line.startsWith("\\")) {
      rows.push({ kind: "meta", number: null, sign: "", text: line });
      continue;
    }
    if (line.startsWith("@@")) {
      const match = /^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/.exec(line);
      if (match) {
        oldLine = Number(match[1]);
        newLine = Number(match[2]);
        inHunk = true;
      }
      rows.push({ kind: "hunk", number: null, sign: "", text: line });
      continue;
    }

    const sign = line[0] ?? " ";
    if (sign === "+") {
      rows.push({ kind: "add", number: inHunk ? newLine++ : null, sign, text: line.slice(1) });
    } else if (sign === "-") {
      rows.push({ kind: "del", number: inHunk ? oldLine++ : null, sign, text: line.slice(1) });
    } else {
      rows.push({
        kind: "ctx",
        number: inHunk ? newLine : null,
        sign: sign === " " ? " " : "",
        text: sign === " " ? line.slice(1) : line,
      });
      if (inHunk) {
        oldLine += 1;
        newLine += 1;
      }
    }
  }
  return rows;
}

/** A numbered, syntax-highlighted file body. */
function CodeView({
  code,
  path,
  startLine = 1,
  className = "",
}: {
  code: string;
  path?: string;
  startLine?: number;
  className?: string;
}) {
  const source = code.replace(/\r\n/g, "\n").replace(/\n$/, "");
  const lineCount = source ? source.split("\n").length : 0;
  const highlighted = useMemo(() => {
    const language = path ? kindOf(path).lang : undefined;
    if (!language || !hljs.getLanguage(language)) return null;
    try {
      return hljs.highlight(source, { language, ignoreIllegals: true }).value;
    } catch {
      return null;
    }
  }, [source, path]);

  return (
    <div className={`tool-code ${className}`.trim()}>
      {lineCount > 0 && (
        <div className="tool-code-gutter" aria-hidden="true">
          {Array.from({ length: lineCount }, (_, index) => (
            <span key={index}>{startLine + index}</span>
          ))}
        </div>
      )}
      <pre className="tool-code-pre" aria-label="Tool file content">
        {highlighted ? (
          <code
            className="hljs"
            // hljs escapes the source before adding its own spans.
            dangerouslySetInnerHTML={{ __html: highlighted }}
          />
        ) : (
          <code>{source || "(empty file)"}</code>
        )}
      </pre>
    </div>
  );
}

/** A unified diff with per-side line numbers and tinted add/remove rows. */
function DiffView({ diff }: { diff: string }) {
  const rows = useMemo(() => parseDiff(diff), [diff]);
  return (
    <div className="tool-diff" aria-label="File diff">
      {rows.map((row, index) => {
        if (row.kind === "meta" || row.kind === "hunk") {
          return (
            <div className={`tool-diff-row is-${row.kind}`} key={`${index}-${row.kind}`}>
              <span className="tool-diff-code">{row.text || " "}</span>
            </div>
          );
        }
        return (
          <div className={`tool-diff-row is-${row.kind}`} key={`${index}-${row.kind}`}>
            <span className="tool-diff-lineno">{row.number ?? ""}</span>
            <span className="tool-diff-sign">{row.sign || " "}</span>
            <span className="tool-diff-code">{row.text || " "}</span>
          </div>
        );
      })}
    </div>
  );
}

function parseReadResult(result: string) {
  const match = /^(.+?) — lines (\d+)-(\d+) of (\d+)\n([\s\S]*)$/.exec(result);
  if (!match) return null;
  return {
    code: match[5],
    startLine: Number(match[2]),
    caption: `Lines ${match[2]}–${match[3]} of ${match[4]}`,
  };
}

/** Plain tool text with http(s) URLs as real links, never as HTML. */
function LinkifiedText({ text }: { text: string }) {
  const parts = useMemo(() => text.split(/(https?:\/\/[^\s<>"']+)/g), [text]);
  return (
    <div className="tool-drawer-text">
      {parts.map((part, index) => {
        if (!/^https?:\/\//.test(part)) {
          return <span key={`${index}-${part}`}>{part}</span>;
        }
        const trimmed = part.replace(/[),.;\]]+$/, "");
        const tail = part.slice(trimmed.length);
        return (
          <span key={`${index}-${part}`}>
            <button
              type="button"
              className="tool-drawer-link"
              onClick={() => openUrl(trimmed)}
              title={trimmed}
            >
              {trimmed}
            </button>
            {tail}
          </span>
        );
      })}
    </div>
  );
}

function Section({
  title,
  children,
  note,
}: {
  title: string;
  children: ReactNode;
  note?: string;
}) {
  return (
    <section className="tool-drawer-section">
      <h3>{title}</h3>
      {children}
      {note && <p className="tool-drawer-note">{note}</p>}
    </section>
  );
}

/**
 * The body of a tool's inline drawer: metadata plus whichever artifact the
 * call produced (diff, written content, replacement, or search/read output).
 */
export function ToolDetailPanel({ tool }: { tool: ToolCall }) {
  const [copied, setCopied] = useState(false);
  const copyValue =
    tool.name === "write_file" && tool.content !== undefined
      ? tool.content
      : (tool.diff ?? tool.content ?? tool.result ?? "");
  const readView =
    tool.name === "read_file" && tool.result ? parseReadResult(tool.result) : null;
  const resultTitle =
    tool.name === "grep" || tool.name === "find_files"
      ? "Results"
      : tool.name === "web_search"
        ? "Sources and snippets"
        : tool.name === "fetch_url"
          ? "Fetched text"
          : readView
            ? "File content"
            : "Output";
  const hasOutput =
    tool.result !== undefined ||
    tool.content !== undefined ||
    tool.diff !== undefined ||
    (tool.old !== undefined && tool.new !== undefined);

  const copy = async () => {
    if (!copyValue) return;
    try {
      await navigator.clipboard.writeText(copyValue);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard unavailable */
    }
  };

  return (
    <div className="tool-drawer">
      <div className="tool-drawer-bar">
        <span className="tool-drawer-kicker">
          {tool.synthetic ? "Debug fixture" : "Tool output"}
        </span>
        <span className={`tool-drawer-status is-${tool.status}`}>{tool.status}</span>
        <button
          type="button"
          className="tool-drawer-copy"
          onClick={() => void copy()}
          disabled={!copyValue}
          title={copied ? "Copied" : "Copy tool details"}
        >
          <Icon name={copied ? "check" : "copy"} size={12} />
          {copied ? "Copied" : "Copy"}
        </button>
      </div>

      <div className="tool-drawer-meta">
        {tool.path && (
          <div className="tool-drawer-meta-line">
            <Icon name="file" size={13} />
            <span className="tool-drawer-label">Path</span>
            <code>{tool.path}</code>
          </div>
        )}
        {tool.action && (
          <div className="tool-drawer-meta-line">
            <span className="tool-drawer-label">Action</span>
            <span>{tool.action}</span>
          </div>
        )}
        {tool.query && (
          <div className="tool-drawer-meta-line">
            <Icon name="search" size={13} />
            <span className="tool-drawer-label">
              {tool.name === "grep" ? "Pattern" : "Query"}
            </span>
            <code>{tool.query}</code>
          </div>
        )}
        {tool.glob && (
          <div className="tool-drawer-meta-line">
            <span className="tool-drawer-label">Glob</span>
            <code>{tool.glob}</code>
          </div>
        )}
        {tool.url && (
          <div className="tool-drawer-meta-line">
            <Icon name="globe" size={13} />
            <span className="tool-drawer-label">URL</span>
            <button className="tool-drawer-link" onClick={() => openUrl(tool.url ?? "")}>
              {tool.url}
            </button>
          </div>
        )}
        {tool.search_url && (
          <div className="tool-drawer-meta-line">
            <Icon name="globe" size={13} />
            <span className="tool-drawer-label">Search page</span>
            <button
              className="tool-drawer-link"
              onClick={() => openUrl(tool.search_url ?? "")}
            >
              Open DuckDuckGo
            </button>
          </div>
        )}
        {(tool.additions !== undefined || tool.deletions !== undefined) && (
          <div className="tool-drawer-stats">
            <span className="is-add">+{tool.additions ?? 0}</span>
            <span className="is-del">−{tool.deletions ?? 0}</span>
          </div>
        )}
      </div>

      {!hasOutput && (
        <p className="tool-drawer-empty">
          {tool.status === "running"
            ? "This tool is still running. Its output will appear here."
            : "No output was captured for this call."}
        </p>
      )}

      {tool.name === "write_file" && tool.content !== undefined && (
        <Section
          title="Written content"
          note={
            tool.content_truncated
              ? "Clipped in this view; the model received the full content."
              : undefined
          }
        >
          <CodeView code={tool.content} path={tool.path} />
        </Section>
      )}

      {tool.diff !== undefined && (tool.name !== "write_file" || !tool.content) && (
        <Section
          title="Diff"
          note={
            tool.diff_truncated
              ? "Clipped in this view; the model received the full diff."
              : undefined
          }
        >
          <DiffView diff={tool.diff} />
        </Section>
      )}

      {tool.old !== undefined && tool.new !== undefined && tool.diff === undefined && (
        <Section
          title="Replacement"
          note={
            tool.replacement_truncated
              ? "Clipped in this view; the model received the full snippets."
              : undefined
          }
        >
          <CodeView code={tool.old} path={tool.path} className="is-removed" />
          <CodeView code={tool.new} path={tool.path} className="is-added" />
        </Section>
      )}

      {tool.result !== undefined && (
        <Section
          title={resultTitle}
          note={
            tool.result_truncated
              ? "Clipped in this view; the model received the full result."
              : undefined
          }
        >
          {readView ? (
            <>
              <div className="tool-code-caption">{readView.caption}</div>
              <CodeView
                code={readView.code}
                path={tool.path}
                startLine={readView.startLine}
              />
            </>
          ) : (
            <LinkifiedText text={tool.result} />
          )}
        </Section>
      )}
    </div>
  );
}
