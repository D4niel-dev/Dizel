import { memo, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { Icon } from "./Icon";
import * as api from "../services/api";
import { useChat } from "../store/chatStore";
import type { SessionSummary } from "../types";

interface SessionRowProps {
  session: SessionSummary;
  active: boolean;
}

/**
 * One chat row: click to open, double-click to rename, hover for a menu.
 *
 * The menu is position:fixed rather than a nested popover because the chat list
 * scrolls — an absolutely-positioned panel would be clipped by the scroll box.
 * Delete arms on the first click and fires on the second.
 */
function SessionRowInner({ session, active }: SessionRowProps) {
  const openSession = useChat((s) => s.openSession);
  const togglePin = useChat((s) => s.togglePin);
  const archiveSession = useChat((s) => s.archiveSession);
  const renameSession = useChat((s) => s.renameSession);
  const removeSession = useChat((s) => s.removeSession);

  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(session.title);
  const [menu, setMenu] = useState<{ x: number; y: number } | null>(null);
  const [confirming, setConfirming] = useState(false);

  const inputRef = useRef<HTMLInputElement>(null);
  const moreRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (editing) {
      inputRef.current?.focus();
      inputRef.current?.select();
    }
  }, [editing]);

  // Close on outside click. Checking the ref keeps the mousedown that opens an
  // item from closing the menu before the click lands. The portal lives at the
  // document root, so focus moves into it explicitly.
  useEffect(() => {
    if (!menu) return;

    menuRef.current?.querySelector<HTMLButtonElement>(".row-menu-item")?.focus();

    const onDown = (event: MouseEvent) => {
      if (menuRef.current?.contains(event.target as Node)) return;
      setMenu(null);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setMenu(null);
        moreRef.current?.focus();
      }
    };

    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [menu]);

  const openMenu = (event: React.MouseEvent<HTMLButtonElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const width = 168;
    const height = 196;
    const below = rect.bottom + 4;
    setConfirming(false);
    setMenu({
      x: Math.max(8, Math.min(rect.right - width, window.innerWidth - width - 8)),
      y:
        below + height > window.innerHeight - 8
          ? Math.max(8, rect.top - height - 4)
          : below,
    });
  };

  const commit = () => {
    const next = draft.trim();
    setEditing(false);
    if (next && next !== session.title) void renameSession(session.id, next);
    else setDraft(session.title);
  };

  const act = (run: () => void) => {
    setMenu(null);
    moreRef.current?.focus();
    run();
  };

  /** Writes the session to a JSON file the user picks. */
  const exportChat = async () => {
    try {
      const data = await api.getSession(session.id);
      const safe = data.title.replace(/[^\w\-. ]+/g, "").trim().slice(0, 48) || "chat";
      const result = await window.dizel?.saveJson(
        `${safe}.json`,
        JSON.stringify(data, null, 2),
      );
      if (result && typeof result === "object" && "error" in result) {
        useChat.setState({ error: result.error });
      }
    } catch (error) {
      useChat.setState({ error: (error as Error).message });
    }
  };

  if (editing) {
    return (
      <div className="sb-row editing">
        <input
          ref={inputRef}
          className="sb-row-input"
          value={draft}
          maxLength={80}
          spellCheck={false}
          onChange={(event) => setDraft(event.target.value)}
          onBlur={commit}
          onKeyDown={(event) => {
            if (event.key === "Enter") commit();
            if (event.key === "Escape") {
              setDraft(session.title);
              setEditing(false);
            }
          }}
        />
      </div>
    );
  }

  return (
    <div className={active ? "sb-row active" : "sb-row"}>
      {session.pinned && (
        <span className="sb-pin" title="Pinned">
          <Icon name="pin" size={10} />
        </span>
      )}

      <button
        className="sb-row-label"
        onClick={() => void openSession(session.id)}
        onDoubleClick={() => {
          setDraft(session.title);
          setEditing(true);
        }}
        title={session.title}
      >
        {session.title}
      </button>

      <span className="sb-row-actions">
        <button
          ref={moreRef}
          className="row-act"
          title="More"
          aria-label="Chat actions"
          aria-haspopup="menu"
          aria-expanded={Boolean(menu)}
          onClick={openMenu}
        >
          <Icon name="more" size={13} />
        </button>
      </span>

      {menu &&
        createPortal(
          <div
            ref={menuRef}
            className="row-menu"
            style={{ left: menu.x, top: menu.y }}
            role="menu"
            onKeyDown={(event) => {
              if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
              event.preventDefault();
              const items = Array.from(
                menuRef.current?.querySelectorAll<HTMLButtonElement>(".row-menu-item") ?? [],
              );
              const index = items.indexOf(document.activeElement as HTMLButtonElement);
              const next =
                event.key === "ArrowDown"
                  ? (index + 1) % items.length
                  : (index - 1 + items.length) % items.length;
              items[next]?.focus();
            }}
          >
            <button
              role="menuitem"
              className="row-menu-item"
              onClick={() =>
                act(() => {
                  setDraft(session.title);
                  setEditing(true);
                })
              }
            >
              Rename
            </button>
            <button role="menuitem" className="row-menu-item" onClick={() => act(() => void togglePin(session.id))}>
              {session.pinned ? "Unpin" : "Pin"}
            </button>
            <button
              role="menuitem"
              className="row-menu-item"
              onClick={() => act(() => void archiveSession(session.id))}
            >
              {session.archived ? "Restore" : "Archive"}
            </button>
            <button role="menuitem" className="row-menu-item" onClick={() => act(() => void exportChat())}>
              Export JSON…
            </button>
            <div className="row-menu-sep" role="separator" />
            <button
              role="menuitem"
              className="row-menu-item danger"
              onClick={() => {
                if (confirming) act(() => void removeSession(session.id));
                else setConfirming(true);
              }}
            >
              {confirming ? "Click again to delete" : "Delete"}
            </button>
          </div>,
          document.body,
        )}
    </div>
  );
}

export const SessionRow = memo(SessionRowInner);
