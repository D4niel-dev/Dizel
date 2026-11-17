import { useId, useState } from "react";

import type { IconName } from "./Icon";
import { Icon } from "./Icon";
import { ToolDetailPanel } from "./ToolDetailPanel";
import type { ToolCall } from "../types";

const SEARCH_TOOLS = new Set(["grep", "search", "find_files", "web_search"]);
const READ_TOOLS = new Set(["read_file", "list_dir", "file_info", "fetch_url"]);
const WRITE_TOOLS = new Set(["write_file", "edit_file"]);

function iconFor(tool: ToolCall): IconName {
  const name = tool.name.toLowerCase();
  if (SEARCH_TOOLS.has(name)) return "search";
  if (READ_TOOLS.has(name)) return "file";
  if (WRITE_TOOLS.has(name)) return "compose";
  return "check";
}

function label(name: string): string {
  return name.replace(/_/g, " ");
}

/** Inline agent-log rows with an expandable result drawer per call. */
export function ToolLog({ tools }: { tools: ToolCall[] }) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const baseId = useId();
  if (tools.length === 0) return null;

  return (
    <div className="tool-log" aria-label="Tool activity">
      {tools.map((tool) => {
        const open = expanded === tool.id;
        const drawerId = `${baseId}-tool-${tool.id}`;
        return (
          <div className={`tool-entry${open ? " is-open" : ""}`} key={tool.id}>
            <button
              type="button"
              className={`tool-row is-${tool.status}`}
              onClick={() => setExpanded(open ? null : tool.id)}
              aria-expanded={open}
              aria-controls={drawerId}
              title={`${open ? "Hide" : "Show"} ${label(tool.name)} details`}
              aria-label={`${open ? "Hide" : "Show"} ${label(tool.name)} tool details: ${tool.detail}`}
            >
              <span className="tool-status" aria-hidden="true">
                {tool.status === "running" ? (
                  <span className="tool-spinner" />
                ) : (
                  <Icon name={tool.status === "failed" ? "close" : iconFor(tool)} size={12} />
                )}
              </span>
              <span className={`tool-name${tool.status === "running" ? " is-sweeping" : ""}`}>
                {tool.status === "running"
                  ? Array.from(label(tool.name)).map((letter, index) => (
                      <span
                        key={index}
                        style={{ animationDelay: `${index * 45}ms` }}
                      >
                        {letter === " " ? "\u00A0" : letter}
                      </span>
                    ))
                  : label(tool.name)}
              </span>
              <span className="tool-detail">{tool.detail}</span>
              {tool.additions !== undefined && tool.deletions !== undefined && (
                <span className="tool-stats">
                  <span className="is-add">+{tool.additions}</span>
                  <span className="is-del">−{tool.deletions}</span>
                </span>
              )}
              <span className="tool-open" aria-hidden="true">
                <Icon name="chevronRight" size={11} />
              </span>
            </button>
            {open && (
              <div id={drawerId} className="tool-drawer-wrap">
                <ToolDetailPanel tool={tool} />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
