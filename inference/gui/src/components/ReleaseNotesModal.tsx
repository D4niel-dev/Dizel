import { useEffect } from "react";
import type { ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { Icon } from "./Icon";
import { useChat } from "../store/chatStore";
import { CHANGELOG, LATEST_VERSION } from "../services/releaseNotes";

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

export function ReleaseNotesModal() {
  const open = useChat((s) => s.releaseNotesOpen);
  const close = useChat((s) => s.closeReleaseNotes);

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") close();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, close]);

  if (!open) return null;

  return (
    <div
      className="modal-backdrop release-notes-backdrop"
      onMouseDown={(event) => event.target === event.currentTarget && close()}
    >
      <div
        className="modal release-notes-modal"
        role="dialog"
        aria-modal="true"
        aria-label="Release notes"
      >
        <div className="modal-head">
          <h2>Release notes</h2>
          {LATEST_VERSION && <span className="release-notes-version">v{LATEST_VERSION}</span>}
          <button className="modal-x" onClick={close} aria-label="Close release notes">
            <Icon name="close" size={13} />
          </button>
        </div>
        <div className="modal-body release-notes-body">
          <div className="md">
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={{ a: ExternalLink }}>
              {CHANGELOG}
            </ReactMarkdown>
          </div>
        </div>
      </div>
    </div>
  );
}