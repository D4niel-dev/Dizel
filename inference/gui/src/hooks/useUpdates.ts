import { useEffect, useState } from "react";

import type { UpdateState } from "../types";

/** Subscribe to the main process's update state, or null outside Electron. */
export function useUpdates(): UpdateState | null {
  const [state, setState] = useState<UpdateState | null>(() =>
    window.dizel?.updates
      ? {
          status: "idle",
          currentVersion: "",
          latestVersion: "",
          notes: "",
          releaseUrl: "",
          progress: 0,
          error: "",
          autoCheck: true,
          canInstall: false,
          dev: false,
        }
      : null,
  );

  useEffect(() => {
    const bridge = window.dizel?.updates;
    if (!bridge) return undefined;

    let alive = true;
    void bridge
      .getState()
      .then((next) => {
        if (alive) setState(next);
      })
      .catch(() => undefined);
    const off = bridge.onStatus((next) => {
      if (alive) setState(next);
    });
    return () => {
      alive = false;
      off?.();
    };
  }, []);

  return state;
}
