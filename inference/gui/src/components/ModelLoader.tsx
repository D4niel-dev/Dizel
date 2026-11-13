import { useEffect, useState } from "react";

import { useChat } from "../store/chatStore";

/**
 * Compact loading strip shown where the greeting sits on a ready welcome.
 *
 * Deliberately not a full-screen splash: the welcome's other blocks (composer,
 * starters, recall) stay in place underneath, so the composer doesn't jump when
 * the model finishes loading.
 *
 * The bar is indeterminate because the backend reports a stage message but no
 * percentage — a fake percentage would be worse than none. The elapsed counter
 * is what actually reveals a stalled load.
 */
export function ModelLoader() {
  const status = useChat((s) => s.status);
  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    const id = window.setInterval(() => setSeconds((n) => n + 1), 1000);
    return () => window.clearInterval(id);
  }, []);

  const load = status?.load;
  const name = load?.display_name || "the model";
  const stage = load?.message || "Starting…";

  return (
    <div className="loader">
      <div className="loader-head">
        <span className="loader-spinner" aria-hidden="true" />
        <h1>Loading {name}</h1>
      </div>
      <p className="loader-stage">{stage}</p>
      <div className="loader-track" aria-hidden="true">
        <span />
      </div>
      <span className="loader-elapsed">{seconds}s</span>
    </div>
  );
}
