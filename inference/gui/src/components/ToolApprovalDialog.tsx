import { useEffect, useState } from "react";

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

  useEffect(() => {
    if (!approval) return;
    setBusy(false);
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        void resolve(false);
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
  }[approval.action] ?? "Write file";

  const decide = async (approved: boolean) => {
    if (busy) return;
    setBusy(true);
    await resolve(approved);
  };

  return (
    <div className="modal-backdrop tool-approval-backdrop">
      <div
        className="modal tool-approval"
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
            This changes the file on disk. Review the diff before approving.
          </p>
        </div>
        <div className="tool-approval-actions">
          <button className="btn ghost" disabled={busy} onClick={() => void decide(false)} autoFocus>
            Reject
          </button>
          <button className="btn primary" disabled={busy} onClick={() => void decide(true)}>
            Approve
          </button>
        </div>
      </div>
    </div>
  );
}
