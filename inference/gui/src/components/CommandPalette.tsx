import { useEffect, useMemo, useRef, useState } from "react";

import {
  buildCommandGroups,
  filterCommands,
  flattenGroups,
  matchPositions,
} from "../services/commands";
import type { CommandEntry } from "../services/commands";
import { useChat } from "../store/chatStore";
import type { SessionSummary } from "../types";

type PaletteItem =
  | { kind: "command"; entry: CommandEntry }
  | { kind: "session"; session: SessionSummary };

const RECENT_CHATS = 6;
const MAX_CHAT_MATCHES = 8;

function Highlight({ text, query }: { text: string; query: string }) {
  const positions = matchPositions(text, query);
  if (positions.length === 0) return <>{text}</>;
  return (
    <>
      {text.split("").map((char, index) =>
        positions.includes(index) ? <b key={index}>{char}</b> : <span key={index}>{char}</span>,
      )}
    </>
  );
}

/**
 * Global Ctrl+K palette: one search box that jumps to a slash command or a
 * recent chat. Commands reuse the composer drawer's catalog and fuzzy matcher;
 * chats match on title. Picking a command runs it, picking a chat opens it.
 */
export function CommandPalette() {
  const open = useChat((s) => s.paletteOpen);
  const close = useChat((s) => s.closePalette);
  const skills = useChat((s) => s.skills);
  const debug = useChat((s) => s.debug);
  const sessions = useChat((s) => s.sessions);
  const runSlash = useChat((s) => s.runSlash);
  const openSession = useChat((s) => s.openSession);

  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const listRef = useRef<HTMLDivElement | null>(null);

  const commandEntries = useMemo(
    () => flattenGroups(filterCommands(buildCommandGroups(skills, debug), query)),
    [skills, debug, query],
  );
  const chatMatches = useMemo(() => {
    const needle = query.trim();
    const matched = sessions.filter(
      (session) =>
        !session.archived && (needle === "" || matchPositions(session.title, needle).length > 0),
    );
    return matched.slice(0, needle === "" ? RECENT_CHATS : MAX_CHAT_MATCHES);
  }, [sessions, query]);

  const items = useMemo<PaletteItem[]>(
    () => [
      ...commandEntries.map((entry) => ({ kind: "command" as const, entry })),
      ...chatMatches.map((session) => ({ kind: "session" as const, session })),
    ],
    [commandEntries, chatMatches],
  );
  const active = items.length ? Math.min(activeIndex, items.length - 1) : 0;

  useEffect(() => {
    if (!open) return;
    setQuery("");
    setActiveIndex(0);
    inputRef.current?.focus();
  }, [open]);

  useEffect(() => {
    const row = listRef.current?.querySelector<HTMLElement>(".command-item.is-active");
    row?.scrollIntoView({ block: "nearest" });
  }, [active]);

  const pick = (item: PaletteItem) => {
    close();
    if (item.kind === "command") void runSlash(`/${item.entry.name}`);
    else void openSession(item.session.id);
  };

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        close();
      } else if (event.key === "ArrowDown") {
        event.preventDefault();
        setActiveIndex((index) => Math.min(index + 1, items.length - 1));
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        setActiveIndex((index) => Math.max(index - 1, 0));
      } else if (event.key === "Enter") {
        event.preventDefault();
        const item = items[active];
        if (item) pick(item);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, items, active, close, runSlash, openSession]);

  if (!open) return null;

  const chatOffset = commandEntries.length;

  return (
    <div className="command-palette-backdrop" onMouseDown={close}>
      <div
        className="command-palette"
        role="dialog"
        aria-label="Command palette"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <input
          ref={inputRef}
          className="command-palette-input"
          type="text"
          placeholder="Search commands and chats…"
          aria-label="Search commands and chats"
          value={query}
          onChange={(event) => {
            setQuery(event.target.value);
            setActiveIndex(0);
          }}
        />

        <div className="command-palette-list" role="listbox" ref={listRef}>
          {commandEntries.length > 0 && <div className="command-group-label">Commands</div>}
          {commandEntries.map((entry, index) => {
            const name = `/${entry.name}`;
            const isActive = index === active;
            return (
              <div
                key={`cmd-${entry.name}`}
                role="option"
                aria-selected={isActive}
                className={isActive ? "command-item is-active" : "command-item"}
                title={`${name} — ${entry.hint}`}
                onMouseMove={() => setActiveIndex(index)}
                onMouseDown={(event) => event.preventDefault()}
                onClick={() => pick({ kind: "command", entry })}
              >
                <span className="command-name">
                  <Highlight text={name} query={query} />
                </span>
                <span className="command-desc">{entry.hint}</span>
              </div>
            );
          })}

          {chatMatches.length > 0 && <div className="command-group-label">Chats</div>}
          {chatMatches.map((session, index) => {
            const itemIndex = chatOffset + index;
            const isActive = itemIndex === active;
            return (
              <div
                key={`chat-${session.id}`}
                role="option"
                aria-selected={isActive}
                className={isActive ? "command-item is-active" : "command-item"}
                title={session.title}
                onMouseMove={() => setActiveIndex(itemIndex)}
                onMouseDown={(event) => event.preventDefault()}
                onClick={() => pick({ kind: "session", session })}
              >
                <span className="command-name">
                  <Highlight text={session.title} query={query} />
                </span>
                <span className="command-desc">{session.preview || "Open chat"}</span>
              </div>
            );
          })}

          {items.length === 0 && <div className="command-palette-empty">No matches</div>}
        </div>

        <div className="command-palette-foot">
          <span>
            <kbd>↑</kbd>
            <kbd>↓</kbd> move
          </span>
          <span>
            <kbd>enter</kbd> open
          </span>
          <span className="command-foot-last">
            <kbd>esc</kbd> close
          </span>
        </div>
      </div>
    </div>
  );
}