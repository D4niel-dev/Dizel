import { useEffect, useState } from "react";

import { Icon } from "./Icon";
import { Logo } from "./Logo";

export function TitleBar() {
  const [maximized, setMaximized] = useState(false);

  useEffect(() => {
    const bridge = window.dizel;
    if (!bridge) return;
    // The maximize event can fire before React mounts, so ask once on mount.
    void bridge.window
      .state()
      .then((state) => setMaximized(state.maximized))
      .catch(() => undefined);
    return bridge.window.onMaximizeChange(setMaximized);
  }, []);

  const win = window.dizel?.window;

  return (
    <header className="titlebar">
      <Logo size={16} />

      <div className="titlebar-spacer" />

      <div className="tb-controls">
        <button className="tb-btn" onClick={() => win?.minimize()} title="Minimize" aria-label="Minimize">
          <Icon name="minimize" size={11} strokeWidth={1.6} />
        </button>
        <button
          className="tb-btn"
          onClick={() => win?.toggleMaximize()}
          title={maximized ? "Restore" : "Maximize"}
          aria-label={maximized ? "Restore" : "Maximize"}
        >
          <Icon name={maximized ? "restore" : "maximize"} size={10} strokeWidth={1.6} />
        </button>
        <button className="tb-btn close" onClick={() => win?.close()} title="Close" aria-label="Close">
          <Icon name="close" size={12} strokeWidth={1.6} />
        </button>
      </div>
    </header>
  );
}
