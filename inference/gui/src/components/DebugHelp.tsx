import { memo, useMemo } from "react";

interface CommandRow {
  command: string;
  description: string;
}

interface CommandGroup {
  title: string;
  rows: CommandRow[];
}

function parseGroups(body: string): CommandGroup[] {
  const groups: CommandGroup[] = [];
  let group: CommandGroup | null = null;

  for (const raw of body.split("\n")) {
    const line = raw.trimEnd();
    if (!line.trim()) continue;

    if (!line.trimStart().startsWith("/")) {
      group = { title: line.trim(), rows: [] };
      groups.push(group);
      continue;
    }

    if (!group) {
      group = { title: "Commands", rows: [] };
      groups.push(group);
    }

    const split = line.search(/\s{2,}/);
    group.rows.push({
      command: (split === -1 ? line : line.slice(0, split)).trim(),
      description: split === -1 ? "" : line.slice(split).trim(),
    });
  }

  return groups;
}

export function countCommands(body: string): number {
  return body.split("\n").filter((line) => line.trimStart().startsWith("/")).length;
}

export const DebugHelp = memo(function DebugHelp({ body }: { body: string }) {
  const groups = useMemo(() => parseGroups(body), [body]);

  return (
    <div className="debug-panel-scroll">
      <p className="debug-help-lead">
        Type a command in the composer. Drag the header to move; <kbd>Esc</kbd> closes.
      </p>
      {groups.map((group) => (
        <section className="debug-cmd-group" key={group.title} aria-label={group.title}>
          <div className="debug-cmd-group-title">{group.title}</div>
          <div className="debug-cmd-list" role="list">
            {group.rows.map((row) => (
              <div className="debug-cmd-row" role="listitem" key={row.command}>
                <code>{row.command}</code>
                <span>{row.description}</span>
              </div>
            ))}
          </div>
        </section>
      ))}
    </div>
  );
});
