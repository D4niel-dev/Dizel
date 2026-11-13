import { useEffect, useState } from "react";

import { setTools, toolsState } from "../services/api";
import type { ToolState } from "../services/api";
import { useChat } from "../store/chatStore";

/**
 * The GUI's counterpart to the TUI's tools menu.
 *
 * It shows one master switch rather than a checkbox per tool: the API gates the
 * whole registry, and a row of switches that all move together would be a lie
 * about how it works.
 */
export function ToolsSheet() {
  const open = useChat((s) => s.toolsOpen);
  const close = useChat((s) => s.closeTools);
  const [state, setState] = useState<ToolState | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!open) return;
    let alive = true;
    void toolsState()
      .then((next) => {
        if (alive) setState(next);
      })
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [open]);

  if (!open) return null;

  const toggle = async () => {
    if (busy) return;
    setBusy(true);
    try {
      setState(await setTools(!state?.enabled));
    } catch {
      /* the panel already shows the last known state */
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="debug-panel tools-sheet" role="dialog" aria-label="Tools">
      <div className="debug-panel-head">
        <span className="debug-dot" />
        <span className="debug-panel-title">Tools</span>
        <button className="debug-panel-close" onClick={close} aria-label="Close tools">
          ×
        </button>
      </div>

      <div className="tools-body">
        <button
          className={state?.enabled ? "tools-switch is-on" : "tools-switch"}
          onClick={() => void toggle()}
          disabled={busy || !state}
        >
          <span className="tools-switch-dot" />
          {state?.enabled ? "Model can call tools" : "Tool calls are off"}
        </button>

        <p className="tools-note">
          {state?.enabled
            ? "Offered on every request — costs prompt tokens."
            : "Off by default. Enabling adds ~330 tokens to each request."}
        </p>

        <div className="tools-list">
          {(state?.available ?? []).map((name) => (
            <span key={name} className="tools-chip">
              {name}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
