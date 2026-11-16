import { useEffect, useRef, useState } from "react";

import { useChat } from "../store/chatStore";
import { Icon } from "./Icon";

function lineClass(line: string): string {
  if (line.startsWith("+++") || line.startsWith("---")) return "is-meta";
  if (line.startsWith("@@")) return "is-hunk";
  if (line.startsWith("+")) return "is-add";
  if (line.startsWith("-")) return "is-del";
  return "";
}

/**
 * A file write is reviewed before it reaches disk. This gate is driven by the
 * server's approval request and posts the decision back over the same sidecar;
 * the SSE tool row then settles from running to done or failed.
 */
export function ToolApprovalDialog() {
  const approval = useChat((s) => s.approval);
  const resolve = useChat((s) => s.resolveApproval);
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false);

  const decide = async (decision: string) => {
    if (busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    try {
      await resolve(decision);
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  };

  useEffect(() => {
    if (!approval) return;
    busyRef.current = false;
    setBusy(false);
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        void decide("deny");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [approval, resolve]);

  if (!approval) return null;

  const action = {
    create: "Create file",
    overwrite: "Overwrite file",
    edit: "Edit file",
    delete: "Delete file",
  }[approval.action] ?? "Write file";
  // Delete is always a per-call decision: this gate never offers a standing
  // "Always allow", and no trust tier may auto-approve it.
  const isDelete = approval.action === "delete";
  const warning = isDelete
    ? "This moves the file to the recycle bin. Every deletion needs your explicit approval — it can never be auto-approved."
    : "This changes the file on disk. Review the diff before approving.";

  return (
    <div className="modal-backdrop tool-approval-backdrop">
      <div
        className={isDelete ? "modal tool-approval is-delete" : "modal tool-approval"}
        role="dialog"
        aria-modal="true"
        aria-label={action}
      >
        <div className="modal-head tool-approval-head">
          <h2>{action}</h2>
          {approval.synthetic && <span className="tool-approval-fixture">debug fixture</span>}
          <span className="tool-approval-kind">{approval.tool}</span>
        </div>
        <div className="tool-approval-body">
          <div className="tool-approval-path">
            <Icon name="file" size={13} />
            <code>{approval.path}</code>
          </div>
          <div className="tool-approval-stats">
            <span className="is-add">+{approval.additions}</span>
            <span className="is-del">−{approval.deletions}</span>
          </div>
          <pre className="tool-approval-diff" aria-label="Proposed file diff">
            {approval.diff.split("\n").map((line, index) => (
              <span className={lineClass(line)} key={`${index}-${line}`}>
                {line || " "}
                {"\n"}
              </span>
            ))}
          </pre>
          <p className="tool-approval-warning">
            {warning}
          </p>
          {!isDelete && (
            <p className="tool-approval-trust-hint">
              "Always allow" stays on until you revoke it in Settings → Chat → File writes.
            </p>
          )}
        </div>
        <div className="tool-approval-actions">
          <button className="btn ghost" disabled={busy} onClick={() => void decide("deny")} autoFocus>
            Reject
          </button>
          {isDelete ? (
            <button className="btn primary" disabled={busy} onClick={() => void decide("allow_once")}>
              Approve
            </button>
          ) : (
            <>
              <button className="btn primary" disabled={busy} onClick={() => void decide("allow_once")}>
                Allow once
              </button>
              <button className="btn ghost" disabled={busy} onClick={() => void decide("always_session")}>
                Always this session
              </button>
              <button className="btn ghost" disabled={busy} onClick={() => void decide("always_persistent")}>
                Always allow
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
