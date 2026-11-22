import { useEffect } from "react";
import { createPortal } from "react-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";

import type { Message } from "../types";
import { Icon } from "./Icon";

/** Every stored version of one reply, side by side. */
export function CompareDialog({
  versions,
  currentIndex,
  onClose,
}: {
  versions: Message[];
  currentIndex: number;
  onClose: () => void;
}) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // The bubble's typing transforms would become the containing block for a
  // fixed overlay, so the compare view mounts on the body instead.
  return createPortal(
    <div
      className="compare-overlay"
      role="dialog"
      aria-modal="true"
      aria-label="Compare versions"
      onClick={onClose}
    >
      <div className="compare-panel" onClick={(event) => event.stopPropagation()}>
        <div className="compare-head">
          <h2>Compare versions</h2>
          <button className="icon-btn" onClick={onClose} aria-label="Close compare">
            <Icon name="close" size={14} />
          </button>
        </div>
        <div className="compare-columns">
          {versions.map((version, index) => (
            <div
              className={index === currentIndex ? "compare-col is-current" : "compare-col"}
              key={version.id}
            >
              <div className="compare-col-head">
                <span>Version {index + 1}</span>
                {index === currentIndex && (
                  <span className="compare-current-tag">Current</span>
                )}
              </div>
              <div className="compare-col-body md">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeHighlight]}
                >
                  {version.content}
                </ReactMarkdown>
              </div>
              {version.tools && version.tools.length > 0 && (
                <div className="compare-tools">
                  {version.tools.length} tool call{version.tools.length === 1 ? "" : "s"}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>,
    document.body,
  );
}
