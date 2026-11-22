import { useEffect, useRef, useState } from "react";

import * as api from "../services/api";
import type { ContextUsage } from "../services/api";
import { useChat } from "../store/chatStore";
import { contextUsage } from "../services/context";
import { usageWindow } from "../services/api";
import type { UsageWindow } from "../services/api";
import { modelName } from "../services/model";
import { ContextPanel } from "./ContextPanel";

export function StatusBar() {
  const status = useChat((s) => s.status);
  const messages = useChat((s) => s.messages);
  const streaming = useChat((s) => s.streaming);
  const booted = useChat((s) => s.booted);
  const sessionId = useChat((s) => s.sessionId);
  const budgetOverride = useChat((s) => s.budgetOverride);
  const verbosity = useChat((s) => s.verbosity);
  const modePrompt = useChat((s) => s.modePrompt);
  const trust = useChat((s) => s.trust);
  const approval = useChat((s) => s.approval);
  const trustActive = Boolean(trust && (trust.persistent || trust.session_trusted));

  // Trust can change from Settings, from a gate decision, or from another
  // surface entirely — refresh on session/approval changes and with the poll.
  useEffect(() => {
    void useChat.getState().refreshTrust();
  }, [sessionId, approval]);
  const [usage, setUsage] = useState<ContextUsage | null>(null);
  const [ctxOpen, setCtxOpen] = useState(false);
  const [ctxAnchor, setCtxAnchor] = useState<{
    left: number;
    top: number;
    width: number;
  } | null>(null);
  const ctxButtonRef = useRef<HTMLButtonElement | null>(null);

  // The panel is a popover, so it closes the way every other one does.
  useEffect(() => {
    if (!ctxOpen) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setCtxOpen(false);
    };
    const onDown = (event: MouseEvent) => {
      const target = event.target as Element | null;
      if (
        !target ||
        (!target.closest(".ctx-anchor") && !target.closest(".ctx-panel"))
      ) {
        setCtxOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    document.addEventListener("mousedown", onDown);
    return () => {
      window.removeEventListener("keydown", onKey);
      document.removeEventListener("mousedown", onDown);
    };
  }, [ctxOpen]);

  // Debounced so a streaming reply doesn't fire a request per token.
  useEffect(() => {
    const timer = window.setTimeout(() => {
      void api
        .contextUsage(sessionId)
        .then(setUsage)
        .catch(() => undefined);
    }, 400);
    return () => window.clearTimeout(timer);
    // Prompt choices (mode, effort, speed, length) change the token count too.
  }, [sessionId, messages.length, streaming, status?.is_ready, verbosity, modePrompt]);

  // The 12-hour usage window. Its whole point is that it *resets*, so the
  // countdown is the useful half of this chip, not the percentage.
  const [window_, setWindow] = useState<UsageWindow | null>(null);
  // Provider spend in dollars, when a price list exists for the active provider.
  // Local checkpoints estimate to null, so this stays quiet for them.
  const [cost, setCost] = useState<{ amount: number; pricing: string } | null>(null);
  useEffect(() => {
    let alive = true;
    const poll = () => {
      void usageWindow()
        .then((next) => {
          if (alive) setWindow(next);
        })
        .catch(() => undefined);
      void api
        .usageReport()
        .then((report) => {
          if (!alive) return;
          const row = report.providers.find((entry) => entry.active);
          const amount = row?.estimated_cost;
          setCost(
            row && typeof amount === "number" && amount > 0
              ? { amount, pricing: row.pricing }
              : null,
          );
        })
        .catch(() => undefined);
      void useChat.getState().refreshTrust();
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

      {trustActive && (
        <>
          <span className="sep">·</span>
          <span
            className="trust-chip"
            title="Writes and edits are auto-approved. Revoke in Settings → Chat → File writes."
          >
            Auto-approving writes
          </span>
        </>
      )}

      <span className="spacer" />

      <span>{messages.length} msg</span>

      {verbosity !== "normal" && (
        <>
          <span className="sep">·</span>
          <span title="Response length — change it in the + menu">
            {verbosity} replies
          </span>
        </>
      )}

      <span className="sep">·</span>
      <span className="ctx-anchor">
        <button
          type="button"
          className="ctx-trigger"
          data-capability="context.rail"
          aria-haspopup="dialog"
          aria-expanded={ctxOpen}
          title={breakdown}
          ref={ctxButtonRef}
          onClick={() => {
            if (ctxOpen) {
              setCtxOpen(false);
              return;
            }
            const rect = ctxButtonRef.current?.getBoundingClientRect();
            setCtxAnchor(
              rect ? { left: rect.left, top: rect.top, width: rect.width } : null,
            );
            setCtxOpen(true);
          }}
        >
          <span className={percent >= 80 ? "ctx-warn" : undefined}>
            {exact ? "" : "~"}
            {used.toLocaleString()} / {contextLength} ctx
          </span>
          <span className="ctx-bar">
            <span className="ctx-fill" style={{ width: `${percent}%` }} />
            {reservedPercent > 0 && (
              <span
                className="ctx-reserved"
                style={{ left: `${percent}%`, width: `${reservedPercent}%` }}
              />
            )}
          </span>
        </button>
        {usage?.task && (
          <span className="ctx-task" title={breakdown}>
            {usage.task}
          </span>
        )}
        {ctxOpen && <ContextPanel usage={usage} anchor={ctxAnchor} />}
      </span>
      {window_ && (
        <>
          <span className="sep">·</span>
          <span
            className={
              window_.exhausted || window_.percent >= 84 ? "ctx-warn" : undefined
            }
            title={
              window_.exhausted
                ? `${window_.provider} usage window spent — resets in ${window_.resets_in}`
                : `${window_.provider} usage window resets in ${window_.resets_in}`
            }
          >
            {window_.used.toLocaleString()} / {window_.capacity} used
            {window_.exhausted ? " · spent" : ` · ${window_.resets_in}`}
          </span>

        </>
      )}

      {cost && (
        <>
          <span className="sep">·</span>
          <span title={`List-price estimate · ${cost.pricing}. Not a bill.`}>
            ~${cost.amount < 0.01 ? cost.amount.toFixed(3) : cost.amount.toFixed(2)}
          </span>
        </>
      )}

    </footer>
  );
}
