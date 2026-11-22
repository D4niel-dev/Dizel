import { useEffect, useRef, useState } from "react";

import { useChat } from "../store/chatStore";
import { Icon } from "./Icon";

/**
 * A plan is reviewed before the model acts on it. This gate is driven by the
 * server's plan event and posts the decision back on the same sidecar.
 */
export function PlanApprovalDialog() {
  const plan = useChat((s) => s.planApproval);
  const resolve = useChat((s) => s.resolvePlan);
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false);

  const decide = async (approved: boolean) => {
    if (busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    try {
      await resolve(approved);
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  };

  useEffect(() => {
    if (!plan) return;
    busyRef.current = false;
    setBusy(false);
  }, [plan]);

  useEffect(() => {
    if (!plan) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      void decide(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [plan, resolve]);

  if (!plan) return null;

  return (
    <div className="modal-backdrop plan-approval-backdrop">
      <div
        className="modal plan-approval"
        role="dialog"
        aria-modal="true"
        aria-label="Plan approval"
      >
        <div className="modal-head plan-approval-head">
          <span className="plan-approval-title">
            <Icon name="listChecks" size={16} />
            <h2>Plan approval</h2>
          </span>
          {plan.synthetic && <span className="tool-approval-fixture">debug fixture</span>}
          <span className="tool-approval-kind">{plan.steps.length} steps</span>
        </div>
        <div className="plan-approval-body">
          {plan.task && (
            <div className="plan-approval-task">
              <span className="plan-approval-task-label">Task</span>
              <p>{plan.task}</p>
            </div>
          )}
          <ol className="plan-approval-steps">
            {plan.steps.map((step, index) => (
              <li key={`${index}-${step}`}>{step}</li>
            ))}
          </ol>
          <p className="plan-approval-warning">
            <Icon name="alert" size={13} />
            <span>The model runs these steps with its tools. Every write still needs your approval.</span>
          </p>
        </div>
        <div className="tool-approval-actions plan-approval-actions">
          <span className="plan-approval-hint">Esc to discard</span>
          <button
            className="btn ghost"
            disabled={busy}
            onClick={() => void decide(false)}
            autoFocus
          >
            Discard
          </button>
          <button className="btn primary" disabled={busy} onClick={() => void decide(true)}>
            Execute plan
          </button>
        </div>
      </div>
    </div>
  );
}
