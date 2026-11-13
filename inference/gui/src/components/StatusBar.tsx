import { useEffect, useState } from "react";

import * as api from "../services/api";
import type { ContextUsage } from "../services/api";
import { useChat } from "../store/chatStore";
import { contextUsage } from "../services/context";
import { usageWindow } from "../services/api";
import type { UsageWindow } from "../services/api";
import { modelName } from "../services/model";

export function StatusBar() {
  const status = useChat((s) => s.status);
  const messages = useChat((s) => s.messages);
  const streaming = useChat((s) => s.streaming);
  const booted = useChat((s) => s.booted);
  const sessionId = useChat((s) => s.sessionId);
  const budgetOverride = useChat((s) => s.budgetOverride);
  const [usage, setUsage] = useState<ContextUsage | null>(null);

  // Debounced so a streaming reply doesn't fire a request per token.
  useEffect(() => {
    const timer = window.setTimeout(() => {
      void api
        .contextUsage(sessionId)
        .then(setUsage)
        .catch(() => undefined);
    }, 400);
    return () => window.clearTimeout(timer);
  }, [sessionId, messages.length, streaming, status?.is_ready]);

  // The 12-hour usage window. Its whole point is that it *resets*, so the
  // countdown is the useful half of this chip, not the percentage.
  const [window_, setWindow] = useState<UsageWindow | null>(null);
  useEffect(() => {
    let alive = true;
    const poll = () => {
      void usageWindow()
        .then((next) => {
          if (alive) setWindow(next);
        })
        .catch(() => undefined);
    };
    poll();
    const timer = window.setInterval(poll, 30_000);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, [messages.length, streaming]);

  const info = status?.model_info ?? {};
  // Prefer the server's numbers; the local estimate is only a first-paint guess.
  const local = contextUsage(messages, info.ctx_len);
  const realUsed = usage?.used ?? local.used;
  const contextLength = usage?.total ?? local.total;
  const used = budgetOverride ?? realUsed;
  const percent =
    budgetOverride !== null
      ? Math.min(100, Math.round((budgetOverride * 100) / contextLength))
      : (usage?.percent ?? local.percent);
  const reserved = usage?.reserved ?? 0;
  const reservedPercent = contextLength
    ? Math.min(100 - percent, Math.round((reserved * 100) / contextLength))
    : 0;
  const exact = usage?.exact ?? false;
  const breakdown = usage
    ? `System ${usage.system} · Conversation ${usage.conversation}` +
      (reserved ? ` · Reserved ${reserved}` : "") +
      ` of ${contextLength}` +
      (usage.task ? `\nClassified as "${usage.task}" — that set the ${reserved}-token reply budget` : "")
    : "Estimated until the model is loaded";

  const device = info.device ? String(info.device).toUpperCase() : "—";
  const loading = status?.load?.status === "loading";
  const failed = status?.load?.status === "error";
  const state = !booted
    ? "Starting"
    : streaming
      ? "Generating"
      : loading
        ? "Loading"
        : failed
          ? "Load failed"
          : status?.is_ready
            ? "Ready"
            : "Idle";
  const busy = !booted || streaming || loading;

  return (
    <footer className="statusbar">
      <span className={`dot ${busy ? "busy" : status?.is_ready ? "ok" : ""}`} />
      <span>{state}</span>

      <span className="sep">·</span>
      <b>{booted ? modelName(status) : "—"}</b>

      {info.params && (
        <>
          <span className="sep">·</span>
          <span>{info.params}</span>
        </>
      )}

      <span className="sep">·</span>
      <span>{device}</span>

      <span className="spacer" />

      <span>{messages.length} msg</span>

      <span className="sep">·</span>
      <span className={percent >= 80 ? "ctx-warn" : undefined} title={breakdown}>
        {exact ? "" : "~"}
        {used.toLocaleString()} / {contextLength} ctx
      </span>
      {usage?.task && (
        <span className="ctx-task" title={breakdown}>
          {usage.task}
        </span>
      )}
      {window_ && (
        <>
          <span className="sep">·</span>
          <span
            className={
              window_.exhausted || window_.percent >= 84 ? "ctx-warn" : undefined
            }
            title={
              window_.exhausted
                ? `Usage window spent — resets in ${window_.resets_in}`
                : `Usage window resets in ${window_.resets_in}`
            }
          >
            {window_.used.toLocaleString()} / {window_.capacity} used
            {window_.exhausted ? " · spent" : ` · ${window_.resets_in}`}
          </span>

        </>
      )}

      <span className="ctx-bar" title={breakdown}>
        <span className="ctx-fill" style={{ width: `${percent}%` }} />
        {reservedPercent > 0 && (
          <span
            className="ctx-reserved"
            style={{ left: `${percent}%`, width: `${reservedPercent}%` }}
          />
        )}
      </span>
    </footer>
  );
}
