import { useEffect } from "react";

import { useChat } from "../store/chatStore";
import { countCommands, DebugHelp } from "./DebugHelp";
import { Icon } from "./Icon";
import { usePanelDrag } from "./usePanelDrag";

const HELP_TITLE = "Debug commands";

/**
 * Transient message shown when a mode changes — currently only debug.
 *
 * Deliberately not a chat message: toasts are UI state, and anything added to the
 * transcript would be saved to the session and fed back to the model.
 */
export function Toast() {
  const toast = useChat((s) => s.toast);
  const overModal = useChat((s) => s.settingsOpen);
  if (!toast) return null;

  return (
    <div
      className={overModal ? "toast over-modal" : "toast"}
      role="status"
      aria-live="polite"
    >
      {toast}
    </div>
  );
}

/**
 * Debug output panel. Slash-command results land here rather than in the chat.
 */
export function DebugPanel() {
  const panel = useChat((s) => s.debugPanel);
  const close = useChat((s) => s.closeDebugPanel);
  const drag = usePanelDrag();

  const isHelp = panel?.title === HELP_TITLE;
  const commandCount = isHelp && panel ? countCommands(panel.body) : 0;

  useEffect(() => {
    if (!panel) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") close();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [panel, close]);

  useEffect(() => {
    if (!panel) drag.cancelDrag();
  }, [panel]);

  if (!panel) return null;

  return (
    <div
      ref={drag.ref}
      className={`debug-panel${drag.pos ? " is-moved" : ""}${
        drag.dragging ? " is-dragging" : ""
      }`}
      style={drag.pos ? { left: drag.pos.x, top: drag.pos.y } : undefined}
      role="dialog"
      aria-label={panel.title}
    >
      <div
        className="debug-panel-head"
        onPointerDown={drag.beginDrag}
        onPointerMove={drag.moveDrag}
        onPointerUp={drag.stopDrag}
        onPointerCancel={drag.stopDrag}
        onLostPointerCapture={drag.stopDrag}
        onDoubleClick={drag.resetPosition}
      >
        <span
          className="debug-grip"
          role="button"
          tabIndex={0}
          aria-label="Move debug panel; arrow keys move, Home resets"
          title="Drag to move · double-click to reset · arrow keys also move"
          onKeyDown={drag.onGripKeyDown}
        >
          <Icon name="more" size={13} />
        </span>
        <span className="debug-panel-title">{panel.title}</span>
        {isHelp && (
          <span className="debug-panel-count" title={`${commandCount} commands`}>
            {commandCount}
          </span>
        )}
        <button className="debug-panel-close" onClick={close} aria-label="Close command panel">
          <Icon name="close" size={12} />
        </button>
      </div>
      {isHelp ? (
        <DebugHelp body={panel.body} />
      ) : (
        <pre className="debug-panel-body">{panel.body}</pre>
      )}
    </div>
  );
}
