import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { Icon } from "./Icon";
import { useChat } from "../store/chatStore";

/**
 * Response length as a row under Images… in the + menu.
 *
 * The flyout is portalled because the popover clips its own overflow, and it
 * carries `data-picker-keep-open` so clicks inside do not close the parent menu.
 */

const OPTIONS = [
  {
    id: "short",
    label: "Short",
    factor: "0.6×",
    note: "Answer only what was asked",
  },
  {
    id: "normal",
    label: "Normal",
    factor: "1.0×",
    note: "The default length",
  },
  {
    id: "detailed",
    label: "Detailed",
    factor: "1.5×",
    note: "Reasoning, examples and caveats",
  },
] as const;

export function VerbosityPicker({ open }: { open: boolean }) {
  const verbosity = useChat((s) => s.verbosity);
  const setVerbosity = useChat((s) => s.setVerbosity);
  const [panelOpen, setPanelOpen] = useState(false);
  const [position, setPosition] = useState({ top: 0, left: 0 });
  const rowRef = useRef<HTMLDivElement>(null);
  const arrowRef = useRef<HTMLButtonElement>(null);
  const closeTimer = useRef<number | null>(null);
  const pinned = useRef(false);

  const current = OPTIONS.find((item) => item.id === verbosity) ?? OPTIONS[1];

  const clearClose = () => {
    if (closeTimer.current !== null) {
      window.clearTimeout(closeTimer.current);
      closeTimer.current = null;
    }
  };
  const scheduleClose = () => {
    if (pinned.current) return;
    clearClose();
    closeTimer.current = window.setTimeout(() => setPanelOpen(false), 150);
  };
  const openPanel = (pin = false) => {
    clearClose();
    if (pin) pinned.current = true;
    const anchor = arrowRef.current ?? rowRef.current;
    if (!anchor) return;
    const rect = anchor.getBoundingClientRect();
    const width = 236;
    const height = 168;
    const left =
      window.innerWidth - rect.right >= width + 12
        ? rect.right + 8
        : Math.max(12, rect.left - width - 8);
    const top = Math.max(12, Math.min(rect.top - 8, window.innerHeight - height));
    setPosition({ top, left });
    setPanelOpen(true);
  };
  const closePanel = () => {
    pinned.current = false;
    setPanelOpen(false);
  };

  useEffect(() => {
    if (!open) {
      pinned.current = false;
      setPanelOpen(false);
    }
  }, [open]);
  useEffect(() => () => clearClose(), []);

  const choose = (id: (typeof OPTIONS)[number]["id"]) => {
    void setVerbosity(id);
    closePanel();
  };

  return (
    <>
      <div
        ref={rowRef}
        className="mode-row"
        data-capability="chat.verbosity"
        onMouseEnter={() => openPanel(false)}
        onMouseLeave={scheduleClose}
      >
        <button
          type="button"
          className="popover-item"
          onClick={() => openPanel(true)}
        >
          <span className="popover-item-main">
            <Icon name="compose" size={14} /> Length
          </span>
          <span className="popover-hint">{current.label}</span>
        </button>
        <button
          ref={arrowRef}
          type="button"
          className="mode-arrow"
          aria-label="Reply length"
          aria-expanded={panelOpen}
          onClick={() => {
            if (panelOpen && pinned.current) {
              closePanel();
              return;
            }
            openPanel(true);
          }}
          onMouseEnter={() => openPanel(false)}
          onMouseLeave={scheduleClose}
        >
          <Icon name="chevronRight" size={12} />
        </button>
      </div>

      {open &&
        panelOpen &&
        createPortal(
          <div
            className="verbosity-panel"
            data-picker-keep-open=""
            style={{ top: position.top, left: position.left }}
            role="group"
            aria-label="Response length"
            onMouseEnter={clearClose}
            onMouseLeave={scheduleClose}
            onKeyDown={(event) => {
              if (event.key === "Escape") {
                event.preventDefault();
                event.stopPropagation();
                closePanel();
              }
            }}
          >
            <div className="verbosity-options">
              {OPTIONS.map((option) => {
                const active = option.id === verbosity;
                return (
                  <button
                    key={option.id}
                    type="button"
                    className={
                      active ? "verbosity-option active" : "verbosity-option"
                    }
                    aria-pressed={active}
                    onClick={() => choose(option.id)}
                  >
                    <span className="verbosity-option-top">
                      <b>{option.label}</b>
                      <span className="verbosity-factor">{option.factor} budget</span>
                      {active && <Icon name="check" size={12} />}
                    </span>
                    <span className="verbosity-note">{option.note}</span>
                  </button>
                );
              })}
            </div>
          </div>,
          document.body,
        )}
    </>
  );
}
