/**
 * Resolves where the Python sidecar is listening.
 *
 * The Electron main process picks a free port and a random launch token, then
 * pushes them to the renderer. If the renderer loads before the sidecar is
 * healthy, we wait for the event rather than failing.
 */

import type { SidecarInfo } from "../types";

const TIMEOUT_MS = 120_000;

let resolved: SidecarInfo | null = null;
let pending: Promise<SidecarInfo> | null = null;

/** Forget the cached address — the port and token change on every restart. */
export function resetSidecar(): void {
  resolved = null;
  pending = null;
}

export function resolveSidecar(): Promise<SidecarInfo> {
  if (resolved) return Promise.resolve(resolved);
  if (pending) return pending;

  pending = new Promise<SidecarInfo>((resolve, reject) => {
    const bridge = window.dizel;
    if (!bridge) {
      reject(new Error("This window is not running inside the Dizel desktop shell."));
      return;
    }

    let settled = false;
    let offReady: () => void = () => {};
    let offError: () => void = () => {};
    let timer = 0;

    const settle = (apply: () => void) => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timer);
      offReady();
      offError();
      apply();
    };

    timer = window.setTimeout(
      () => settle(() => reject(new Error("The local engine did not start in time."))),
      TIMEOUT_MS,
    );

    offReady = bridge.onSidecarReady((info) => {
      resolved = info;
      settle(() => resolve(info));
    });
    offError = bridge.onSidecarError((message) => {
      settle(() => reject(new Error(message)));
    });

    // The sidecar may already be up if the window reloaded after boot.
    void bridge.getSidecar().then((info) => {
      if (!info) return;
      resolved = info;
      settle(() => resolve(info));
    });
  }).catch((error: unknown) => {
    pending = null;
    throw error;
  });

  return pending;
}
