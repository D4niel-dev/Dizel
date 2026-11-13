import { useEffect, useRef } from "react";

import { matchPositions } from "../services/commands";
import type { CommandEntry, CommandGroup } from "../services/commands";

interface CommandDrawerProps {
  groups: CommandGroup[];
  query: string;
  activeIndex: number;
  count: number;
  onHover: (index: number) => void;
  onPick: (entry: CommandEntry) => void;
}

/**
 * Command tray attached to the composer while a slash prefix is typed. It is
 * an overlay layer, not part of the composer's flow; the textarea keeps focus
 * and the keyboard moves through the options.
 */
export function CommandDrawer({
  groups,
  query,
  activeIndex,
  count,
  onHover,
  onPick,
}: CommandDrawerProps) {
  const listRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const active = listRef.current?.querySelector<HTMLElement>(".command-item.is-active");
    active?.scrollIntoView({ block: "nearest" });
  }, [activeIndex]);

  // A narrower query is a new result set; start it from the top.
  useEffect(() => {
    listRef.current?.scrollTo({ top: 0 });
  }, [query]);

  let index = -1;

  return (
    <div className="command-drawer" id="command-drawer" aria-label="Slash commands">
      <div className="command-drawer-head">
        <span className="command-drawer-title">Commands</span>
        <span className="command-drawer-count">
          {count} {count === 1 ? "command" : "commands"}
        </span>
      </div>

      <div className="command-list" role="listbox" ref={listRef}>
        {groups.map((group) => (
          <div className="command-group" key={group.label}>
            <div className="command-group-label">
              {group.label}
              {group.showCount !== false && (
                <span className="command-group-count">{group.entries.length}</span>
              )}
            </div>
            {group.entries.map((entry) => {
              index += 1;
              const itemIndex = index;
              const active = itemIndex === activeIndex;
              const name = `/${entry.name}`;
              const positions = matchPositions(name, query);
              const nameMatch = positions.length > 0;
              const aliasMatch =
                query && !nameMatch
                  ? entry.aliases.find((alias) => matchPositions(alias, query).length > 0)
                  : null;
              const aliases = entry.aliases.length
                ? ` · aliases: ${entry.aliases.map((alias) => `/${alias}`).join(", ")}`
                : "";

              return (
                <div
                  key={`${group.label}-${entry.name}`}
                  id={`command-option-${entry.name}`}
                  role="option"
                  aria-selected={active}
                  className={active ? "command-item is-active" : "command-item"}
                  title={`${name} — ${entry.hint}${aliases}`}
                  onMouseMove={() => onHover(itemIndex)}
                  onMouseDown={(event) => event.preventDefault()}
                  onClick={() => onPick(entry)}
                >
                  <span className="command-name">
                    {nameMatch
                      ? name.split("").map((char, index) =>
                          positions.includes(index) ? (
                            <b key={index}>{char}</b>
                          ) : (
                            <span key={index}>{char}</span>
                          ),
                        )
                      : name}
                  </span>
                  {aliasMatch && <span className="command-alias">/{aliasMatch}</span>}
                  <span className="command-desc">{entry.hint}</span>
                  {entry.source && entry.source !== "builtin" && (
                    <span className="command-source">
                      {entry.source === "project" ? "local" : entry.source}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        ))}
      </div>

      <div className="command-drawer-foot">
        <span>
          <kbd>↑</kbd>
          <kbd>↓</kbd> move
        </span>
        <span>
          <kbd>tab</kbd> complete
        </span>
        <span>
          <kbd>enter</kbd> run
        </span>
        <span className="command-foot-last">
          <kbd>esc</kbd> close
        </span>
      </div>
    </div>
  );
}
