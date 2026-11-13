import { useEffect, useMemo, useRef, useState } from "react";
import hljs from "highlight.js/lib/common";

import { Icon } from "./Icon";
import type { IconName } from "./Icon";
import { useChat } from "../store/chatStore";

/** Bytes → "12.4 KB". Small numbers stay in bytes so "812 B" does not read as "0 KB". */
export function formatBytes(size: number): string {
  if (!Number.isFinite(size) || size <= 0) return "0 B";
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

interface Kind {
  label: string;
  icon: IconName;
  lang?: string;
}

/** Extension → what the panel calls it, and how to highlight it. */
const KINDS: Record<string, Kind> = {
  ts: { label: "TypeScript", icon: "code", lang: "typescript" },
  tsx: { label: "TypeScript JSX", icon: "code", lang: "typescript" },
  js: { label: "JavaScript", icon: "code", lang: "javascript" },
  jsx: { label: "JavaScript JSX", icon: "code", lang: "javascript" },
  mjs: { label: "JavaScript", icon: "code", lang: "javascript" },
  cjs: { label: "JavaScript", icon: "code", lang: "javascript" },
  py: { label: "Python", icon: "code", lang: "python" },
  rs: { label: "Rust", icon: "code", lang: "rust" },
  go: { label: "Go", icon: "code", lang: "go" },
  java: { label: "Java", icon: "code", lang: "java" },
  c: { label: "C", icon: "code", lang: "c" },
  h: { label: "C header", icon: "code", lang: "c" },
  cpp: { label: "C++", icon: "code", lang: "cpp" },
  cs: { label: "C#", icon: "code", lang: "csharp" },
  rb: { label: "Ruby", icon: "code", lang: "ruby" },
  php: { label: "PHP", icon: "code", lang: "php" },
  swift: { label: "Swift", icon: "code", lang: "swift" },
  kt: { label: "Kotlin", icon: "code", lang: "kotlin" },
  sh: { label: "Shell", icon: "code", lang: "bash" },
  bash: { label: "Shell", icon: "code", lang: "bash" },
  ps1: { label: "PowerShell", icon: "code", lang: "powershell" },
  cmd: { label: "Batch", icon: "code", lang: "dos" },
  sql: { label: "SQL", icon: "code", lang: "sql" },
  html: { label: "HTML", icon: "code", lang: "xml" },
  css: { label: "CSS", icon: "code", lang: "css" },
  scss: { label: "SCSS", icon: "code", lang: "scss" },
  json: { label: "JSON", icon: "code", lang: "json" },
  yaml: { label: "YAML", icon: "code", lang: "yaml" },
  yml: { label: "YAML", icon: "code", lang: "yaml" },
  toml: { label: "TOML", icon: "code", lang: "ini" },
  ini: { label: "INI", icon: "code", lang: "ini" },
  md: { label: "Markdown", icon: "book", lang: "markdown" },
  markdown: { label: "Markdown", icon: "book", lang: "markdown" },
  txt: { label: "Text", icon: "file" },
  log: { label: "Log", icon: "file" },
  csv: { label: "CSV", icon: "file" },
  png: { label: "PNG image", icon: "image" },
  jpg: { label: "JPEG image", icon: "image" },
  jpeg: { label: "JPEG image", icon: "image" },
  gif: { label: "GIF image", icon: "image" },
  webp: { label: "WebP image", icon: "image" },
  bmp: { label: "Bitmap", icon: "image" },
  svg: { label: "SVG", icon: "image" },
};

export function kindOf(name: string): Kind {
  const ext = name.split(".").pop()?.toLowerCase() ?? "";
  return KINDS[ext] ?? { label: ext ? `${ext.toUpperCase()} file` : "File", icon: "file" };
}

/** A NUL byte in the first block is the usual "this is not text" tell. */
function looksBinary(text: string): boolean {
  return text.slice(0, 2000).includes("\u0000");
}

const MIN_WIDTH = 300;
const MAX_WIDTH = 720;

/**
 * The side panel for one attached file.
 *
 * A file attached to a prompt is folded into the text the model receives, so by
 * the time you are reading the transcript the contents are gone from view. This
 * is where they come back: the exact text that was sent, not a re-read of the
 * file from disk — the file may have changed since.
 */
export function AttachmentPreview() {
  const preview = useChat((s) => s.preview);
  const setPreview = useChat((s) => s.setPreview);

  const [wrap, setWrap] = useState(false);
  const [copied, setCopied] = useState<"content" | "path" | null>(null);
  const [width, setWidth] = useState(() => {
    const stored = Number(localStorage.getItem("dizel.panelWidth"));
    return stored >= MIN_WIDTH && stored <= MAX_WIDTH ? stored : 380;
  });
  const [zoom, setZoom] = useState<number | null>(null);
  const [dimensions, setDimensions] = useState<{ w: number; h: number } | null>(null);

  const bodyRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ startX: number; startWidth: number } | null>(null);

  // A new file starts fresh: no leftover zoom or scroll from the last one.
  useEffect(() => {
    setZoom(null);
    setDimensions(null);
    setWrap(false);
    if (bodyRef.current) bodyRef.current.scrollTop = 0;
  }, [preview?.id]);

  useEffect(() => {
    if (!preview) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setPreview(null);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [preview, setPreview]);

  // Drag the left edge to resize. Persisted so the panel keeps its width.
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
    localStorage.setItem("dizel.panelWidth", String(width));
  }, [width]);

  const kind = preview ? kindOf(preview.name) : null;
  const isImage = Boolean(preview?.previewUrl);

  const lines = useMemo(
    () => (preview && !isImage ? (preview.content || "").split("\n") : []),
    [preview, isImage],
  );

  const highlighted = useMemo(() => {
    if (!preview || isImage || !kind?.lang) return null;
    const text = preview.content ?? "";
    if (!text || looksBinary(text)) return null;
    try {
      if (!hljs.getLanguage(kind.lang)) return null;
      return hljs.highlight(text, { language: kind.lang, ignoreIllegals: true }).value;
    } catch {
      return null;
    }
  }, [preview, isImage, kind?.lang]);

  if (!preview || !kind) return null;

  const binary = !isImage && looksBinary(preview.content ?? "");

  const copy = async (value: string, which: "content" | "path") => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(which);
      window.setTimeout(() => setCopied(null), 1400);
    } catch {
      /* clipboard unavailable */
    }
  };

  const stepZoom = (delta: number) =>
    setZoom((current) => {
      const base = current ?? 1;
      return Math.min(4, Math.max(0.25, Math.round((base + delta) * 100) / 100));
    });

  return (
    <aside
      className="att-panel"
      style={{ flexBasis: width }}
      aria-label={`Preview of ${preview.name}`}
    >
      <div
        className="att-resize"
        onMouseDown={(event) => {
          dragRef.current = { startX: event.clientX, startWidth: width };
          document.body.classList.add("is-resizing-panel");
          event.preventDefault();
        }}
        role="separator"
        aria-label="Resize preview"
      />

      <header className="att-panel-head">
        <span className={`att-kind is-${isImage ? "image" : binary ? "binary" : "text"}`}>
          <Icon name={kind.icon} size={13} />
        </span>

        <div className="att-panel-title">
          <span className="att-panel-name" title={preview.path}>
            {preview.name}
          </span>
          <span className="att-panel-meta">
            {kind.label} · {formatBytes(preview.size)}
            {isImage
              ? dimensions
                ? ` · ${dimensions.w}×${dimensions.h}`
                : ""
              : ` · ${lines.length} ${lines.length === 1 ? "line" : "lines"}`}
            {preview.truncated ? " · truncated" : ""}
          </span>
        </div>

        <div className="att-panel-tools">
          {!isImage && !binary && (
            <button
              className={wrap ? "att-tool is-on" : "att-tool"}
              onClick={() => setWrap((value) => !value)}
              title={wrap ? "Stop wrapping long lines" : "Wrap long lines"}
            >
              <Icon name="panel" size={12} />
            </button>
          )}
          <button
            className="att-tool"
            onClick={() => void copy(preview.path, "path")}
            title="Copy file path"
          >
            <Icon name={copied === "path" ? "check" : "folder"} size={12} />
          </button>
          <button
            className="att-tool"
            onClick={() => void window.dizel?.revealInFolder(preview.path)}
            title="Show in folder"
          >
            <Icon name="search" size={12} />
          </button>
          <button
            className="att-tool"
            onClick={() => void window.dizel?.openPath(preview.path)}
            title="Open in its default app"
          >
            <Icon name="compose" size={12} />
          </button>
          <button
            className="modal-x"
            onClick={() => setPreview(null)}
            aria-label="Close preview"
          >
            <Icon name="close" size={13} />
          </button>
        </div>
      </header>

      <div className="att-panel-body-wrap" ref={bodyRef}>
        {isImage ? (
          <div className="att-image-stage">
            <img
              src={preview.previewUrl}
              alt={preview.name}
              style={zoom ? { width: `${Math.round(zoom * 100)}%` } : undefined}
              onLoad={(event) =>
                setDimensions({
                  w: event.currentTarget.naturalWidth,
                  h: event.currentTarget.naturalHeight,
                })
              }
            />
          </div>
        ) : binary ? (
          <div className="att-empty">
            <Icon name="alert" size={16} />
            <strong>Binary file</strong>
            <span>There is no text preview for this type.</span>
          </div>
        ) : (
          <div className={wrap ? "att-code-wrap is-wrapped" : "att-code-wrap"}>
            {!wrap && (
              <div className="att-gutter" aria-hidden="true">
                {lines.map((_, index) => (
                  <span key={index}>{index + 1}</span>
                ))}
              </div>
            )}
            <pre className="att-code">
              {highlighted ? (
                <code
                  className="hljs"
                  // hljs escapes the source before adding its own spans, so this
                  // is its output rather than the raw file.
                  dangerouslySetInnerHTML={{ __html: highlighted }}
                />
              ) : (
                <code>{preview.content || "(empty file)"}</code>
              )}
            </pre>
          </div>
        )}
      </div>

      {isImage ? (
        <footer className="att-zoom-bar">
          <button className="att-tool" onClick={() => stepZoom(-0.25)} title="Zoom out">
            <Icon name="minimize" size={12} />
          </button>
          <span className="att-zoom-value">
            {zoom ? `${Math.round(zoom * 100)}%` : "Fit"}
          </span>
          <button className="att-tool" onClick={() => stepZoom(0.25)} title="Zoom in">
            <Icon name="plus" size={12} />
          </button>
          <button className="att-tool" onClick={() => setZoom(null)} title="Fit to panel">
            <Icon name="maximize" size={12} />
          </button>
          <span className="spacer" />
          <span className="att-zoom-note">Not sent to the model — no vision.</span>
        </footer>
      ) : (
        <footer className="att-panel-foot">
          <span>
            {binary
              ? "Not text, so the model received a note rather than the contents."
              : preview.truncated
                ? "Only the first part was sent — the model saw this much."
                : `${lines.length} lines sent to the model.`}
          </span>
          <button
            className="att-foot-copy"
            onClick={() => void copy(preview.content, "content")}
            disabled={binary}
          >
            <Icon name={copied === "content" ? "check" : "copy"} size={11} />
            {copied === "content" ? "Copied" : "Copy"}
          </button>
        </footer>
      )}
    </aside>
  );
}
