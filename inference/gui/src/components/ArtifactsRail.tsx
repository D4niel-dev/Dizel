import { useEffect, useMemo, useRef, useState } from "react";

import { Icon } from "./Icon";
import { CodeView, ToolDetailPanel } from "./ToolDetailPanel";
import { kindOf } from "./AttachmentPreview";
import { useChat } from "../store/chatStore";
import {
  artifactVerb,
  collectArtifacts,
  fileName,
  relativeTime,
  type CodeArtifact,
  type FileArtifact,
} from "../services/artifacts";

const MIN_WIDTH = 320;
const MAX_WIDTH = 760;
const WIDTH_KEY = "dizel.artifactsWidth";

type Tab = "files" | "code";

function CopyButton({ value, label }: { value: string; label: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      /* clipboard unavailable */
    }
  };
  return (
    <button
      className="art-tool"
      onClick={() => void copy()}
      title={copied ? "Copied" : label}
      aria-label={label}
    >
      <Icon name={copied ? "check" : "copy"} size={12} />
    </button>
  );
}

function FileRow({ artifact, now }: { artifact: FileArtifact; now: number }) {
  const [open, setOpen] = useState(false);
  const kind = kindOf(artifact.path);
  const tool = artifact.tool;
  const added = tool.additions ?? 0;
  const removed = tool.deletions ?? 0;

  return (
    <li className={open ? "art-row is-open" : "art-row"}>
      <button
        className="art-row-head"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        <span className={`art-kind is-${tool.status}`}>
          <Icon name={kind.icon} size={13} />
        </span>
        <span className="art-row-text">
          <span className="art-row-name" title={artifact.path}>
            {fileName(artifact.path)}
          </span>
          <span className="art-row-meta">
            {artifactVerb(tool)} · {relativeTime(artifact.createdAt, now)}
          </span>
        </span>
        {(added > 0 || removed > 0) && (
          <span className="art-stats">
            {added > 0 && <span className="is-add">+{added}</span>}
            {removed > 0 && <span className="is-del">−{removed}</span>}
          </span>
        )}
        <Icon name={open ? "chevronUp" : "chevronDown"} size={12} />
      </button>
      {open && (
        <div className="art-row-body">
          <ToolDetailPanel tool={tool} />
        </div>
      )}
    </li>
  );
}

function CodeCard({ artifact }: { artifact: CodeArtifact }) {
  const [open, setOpen] = useState(false);
  const lines = artifact.code.split("\n").length;
  const firstLine = artifact.code.split("\n").find((line) => line.trim()) ?? "";

  return (
    <li className={open ? "art-code is-open" : "art-code"}>
      <div className="art-code-head">
        <button
          className="art-code-toggle"
          aria-expanded={open}
          onClick={() => setOpen((value) => !value)}
        >
          <span className="art-code-lang">{artifact.language}</span>
          <span className="art-code-folded" title={firstLine}>
            {firstLine.trim()}
          </span>
          <span className="art-code-count">{lines} ln</span>
          <Icon name={open ? "chevronUp" : "chevronDown"} size={12} />
        </button>
        <CopyButton value={artifact.code} label="Copy block" />
      </div>
      {open && (
        <div className="art-code-body">
          <CodeView
            code={artifact.code}
            path={artifact.language === "text" ? undefined : `snippet.${artifact.language}`}
          />
        </div>
      )}
    </li>
  );
}

/**
 * What this conversation produced.
 *
 * The desktop app had no answer to "where did that file go": a write, an edit
 * and a diff were each buried in the turn that caused them, and a code block
 * was only ever in the middle of a wall of prose. This rail collects them, so
 * the terminal's Ctrl+A has an equivalent here.
 */
export function ArtifactsRail() {
  const open = useChat((s) => s.artifactsOpen);
  const close = useChat((s) => s.closeArtifacts);
  const messages = useChat((s) => s.messages);
  const [tab, setTab] = useState<Tab>("files");
  const [width, setWidth] = useState(() => {
    const stored = Number(localStorage.getItem(WIDTH_KEY));
    return stored >= MIN_WIDTH && stored <= MAX_WIDTH ? stored : 360;
  });
  const [now, setNow] = useState(Date.now());
  const dragRef = useRef<{ startX: number; startWidth: number } | null>(null);

  const artifacts = useMemo(() => collectArtifacts(messages), [messages]);
  const files = artifacts.files;
  const code = artifacts.code;
  const total = files.length + code.length;

  // Relative times age while the panel is open; a minute is fine.
  useEffect(() => {
    if (!open) return undefined;
    const id = window.setInterval(() => setNow(Date.now()), 60_000);
    return () => window.clearInterval(id);
  }, [open]);

  useEffect(() => {
    if (!open) return undefined;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") close();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, close]);

  useEffect(() => {
    const onMove = (event: MouseEvent) => {
      const drag = dragRef.current;
      if (!drag) return;
      setWidth(
        Math.min(
          MAX_WIDTH,
          Math.max(MIN_WIDTH, drag.startWidth + (drag.startX - event.clientX)),
        ),
      );
    };
    const onUp = () => {
      if (!dragRef.current) return;
      dragRef.current = null;
      document.body.classList.remove("is-resizing-panel");
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, []);

  useEffect(() => {
    localStorage.setItem(WIDTH_KEY, String(width));
  }, [width]);

  // A file-backed conversation lands on Files; a chat that only wrote code
  // should not open onto an empty tab.
  useEffect(() => {
    if (!open) return;
    if (files.length === 0 && code.length > 0) setTab("code");
  }, [open, files.length, code.length]);

  if (!open) return null;

  const shown = tab === "files" ? files.length : code.length;

  return (
    <aside
      className="art-panel"
      style={{ flexBasis: width }}
      aria-label="Artifacts produced in this conversation"
    >
      <div
        className="att-resize"
        onMouseDown={(event) => {
          dragRef.current = { startX: event.clientX, startWidth: width };
          document.body.classList.add("is-resizing-panel");
          event.preventDefault();
        }}
        role="separator"
        aria-label="Resize artifacts"
      />

      <header className="art-head">
        <span className="art-title">
          Artifacts
          <span className="art-count">{total}</span>
        </span>
        <div className="art-tabs" role="tablist" aria-label="Artifact kinds">
          <button
            role="tab"
            aria-selected={tab === "files"}
            className={tab === "files" ? "art-tab is-on" : "art-tab"}
            onClick={() => setTab("files")}
          >
            Files
            <span>{files.length}</span>
          </button>
          <button
            role="tab"
            aria-selected={tab === "code"}
            className={tab === "code" ? "art-tab is-on" : "art-tab"}
            onClick={() => setTab("code")}
          >
            Code
            <span>{code.length}</span>
          </button>
        </div>
        <button className="modal-x" onClick={close} aria-label="Close artifacts">
          <Icon name="close" size={13} />
        </button>
      </header>

      <div className="art-body">
        {shown === 0 ? (
          <div className="art-empty">
            <Icon name={tab === "files" ? "folder" : "code"} size={18} />
            <strong>{tab === "files" ? "No files yet" : "No code yet"}</strong>
            <span>
              {tab === "files"
                ? "Files Dizel writes or edits land here, with the diff it applied."
                : "Fenced code blocks from finished replies are collected here."}
            </span>
          </div>
        ) : (
          <ul className="art-list">
            {tab === "files"
              ? files.map((artifact) => (
                  <FileRow artifact={artifact} now={now} key={artifact.key} />
                ))
              : code.map((artifact) => (
                  <CodeCard artifact={artifact} key={artifact.key} />
                ))}
          </ul>
        )}
      </div>
    </aside>
  );
}
