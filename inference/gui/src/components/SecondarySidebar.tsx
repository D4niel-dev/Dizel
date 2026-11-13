import { useEffect, useMemo, useState } from "react";

import { Icon } from "./Icon";
import { SessionRow } from "./SessionRow";
import { useChat } from "../store/chatStore";
import type { SystemStats, WorkspaceView } from "../types";

const TITLES: Record<WorkspaceView, string> = {
  Chats: "Chats",
  Image: "Images",
  Presentation: "Presentations",
  Riset: "Riset",
  Archived: "Archived",
  Library: "Library",
  Monitor: "Monitor",
};

function Empty({
  icon,
  title,
  body,
  hint,
}: {
  icon: Parameters<typeof Icon>[0]["name"];
  title: string;
  body: string;
  hint?: string;
}) {
  return (
    <div className="sub-empty">
      <span className="sub-empty-icon">
        <Icon name={icon} size={18} />
      </span>
      <div className="sub-empty-title">{title}</div>
      <div className="sub-empty-body">{body}</div>
      {hint && <div className="sub-empty-hint">{hint}</div>}
    </div>
  );
}

/** A 0–100 reading with a colour that shifts as it fills. */
function Meter({ value }: { value: number }) {
  const clamped = Math.max(0, Math.min(100, value));
  const tone = clamped >= 85 ? "hot" : clamped >= 60 ? "warm" : "cool";
  return (
    <div className="meter" aria-hidden="true">
      <span data-tone={tone} style={{ width: `${clamped}%` }} />
    </div>
  );
}

/** Buckets a session's timestamp into a human group label. */
function bucket(created: string): string {
  const parsed = new Date(created.replace(" ", "T"));
  if (Number.isNaN(parsed.getTime())) return "Older";

  const now = new Date();
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const stamp = parsed.getTime();

  if (stamp >= startOfToday) return "Today";
  if (stamp >= startOfToday - 86_400_000) return "Yesterday";
  if (stamp >= startOfToday - 6 * 86_400_000) return "This week";
  return "Older";
}

const GROUP_ORDER = ["Pinned", "Today", "Yesterday", "This week", "Older"];

function ArchivedView() {
  const sessions = useChat((s) => s.sessions);
  const sessionId = useChat((s) => s.sessionId);
  const archived = sessions.filter((session) => session.archived);

  if (archived.length === 0) {
    return (
      <Empty
        icon="archive"
        title="Nothing archived"
        body="Archived chats leave the sidebar but stay on disk."
        hint="Use a chat's ⋯ menu and choose Archive."
      />
    );
  }

  return (
    <>
      <div className="sub-intro">
        {archived.length} archived {archived.length === 1 ? "chat" : "chats"} — restoring
        returns them to the sidebar.
      </div>
      <div className="sub-list">
        {archived.map((session) => (
          <SessionRow key={session.id} session={session} active={session.id === sessionId} />
        ))}
      </div>
    </>
  );
}

function ChatsView() {
  const sessions = useChat((s) => s.sessions);
  const sessionId = useChat((s) => s.sessionId);
  const importChat = useChat((s) => s.importChat);
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return sessions;
    return sessions.filter((session) =>
      `${session.title} ${session.preview}`.toLowerCase().includes(needle),
    );
  }, [sessions, query]);

  const groups = useMemo(() => {
    const map = new Map<string, typeof sessions>();
    for (const session of filtered) {
      const key = session.pinned ? "Pinned" : bucket(session.created);
      const list = map.get(key);
      if (list) list.push(session);
      else map.set(key, [session]);
    }
    return GROUP_ORDER.filter((label) => map.has(label)).map(
      (label) => [label, map.get(label) as typeof sessions] as const,
    );
  }, [filtered]);

  return (
    <>
      <div className="sub-search">
        <Icon name="search" size={13} />
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search chats…"
          spellCheck={false}
        />
        <button className="sub-icon-btn" onClick={() => void importChat()} title="Import a chat JSON">
          <Icon name="arrowUp" size={13} style={{ transform: "rotate(180deg)" }} />
        </button>
      </div>

      <div className="sub-list">
        {groups.map(([label, items]) => (
          <div key={label}>
            <div className="sb-sub">{label}</div>
            {items.map((session) => (
              <SessionRow key={session.id} session={session} active={session.id === sessionId} />
            ))}
          </div>
        ))}
        {filtered.length === 0 && <div className="sub-hint">No chats match.</div>}
      </div>
    </>
  );
}

function RisetView() {
  const sessions = useChat((s) => s.sessions);
  const prompts = useChat((s) => s.prompts);
  const openSession = useChat((s) => s.openSession);
  const [query, setQuery] = useState("");

  const needle = query.trim().toLowerCase();
  const chatHits = needle
    ? sessions.filter((s) => `${s.title} ${s.preview}`.toLowerCase().includes(needle))
    : [];
  const promptHits = needle
    ? prompts.filter((p) => `${p.title} ${p.body}`.toLowerCase().includes(needle))
    : [];

  return (
    <>
      <div className="sub-search">
        <Icon name="search" size={13} />
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search everything…"
          spellCheck={false}
          autoFocus
        />
      </div>

      {!needle && (
        <Empty
          icon="search"
          title="Global search"
          body="Searches chat titles, previews and saved prompts. Full-text search over message bodies lands with the archive view."
        />
      )}

      {needle && (
        <div className="sub-list">
          {chatHits.length > 0 && <div className="sub-group">Chats · {chatHits.length}</div>}
          {chatHits.map((session) => (
            <button
              key={session.id}
              className="sub-row"
              onClick={() => void openSession(session.id)}
              title={session.title}
            >
              {session.title}
            </button>
          ))}

          {promptHits.length > 0 && <div className="sub-group">Prompts · {promptHits.length}</div>}
          {promptHits.map((prompt) => (
            <div key={prompt.id} className="sub-row static" title={prompt.body}>
              {prompt.title}
            </div>
          ))}

          {chatHits.length + promptHits.length === 0 && (
            <div className="sub-hint">Nothing matched “{query}”.</div>
          )}
        </div>
      )}
    </>
  );
}

function LibraryView() {
  const prompts = useChat((s) => s.prompts);
  const addPrompt = useChat((s) => s.addPrompt);
  const removePrompt = useChat((s) => s.removePrompt);
  const setDraft = useChat((s) => s.setDraft);

  const [open, setOpen] = useState(false);
  const [title, setTitle] = useState("");
  const [body, setBody] = useState("");
  const [removing, setRemoving] = useState<Set<string>>(new Set());

  const removeWithMotion = (id: string) => {
    setRemoving((current) => new Set(current).add(id));
    window.setTimeout(() => {
      removePrompt(id);
      setRemoving((current) => {
        const next = new Set(current);
        next.delete(id);
        return next;
      });
    }, 200);
  };

  return (
    <>
      <div className="sub-intro">
        Save your best prompts, instructions and persona structures here.
      </div>

      <button
        className="sub-primary"
        onClick={() => setOpen((value) => !value)}
      >
        <Icon name="plus" size={13} />
        New Prompt
      </button>

      {open && (
        <div className="sub-form">
          <input
            value={title}
            onChange={(event) => setTitle(event.target.value)}
            placeholder="Title"
            spellCheck={false}
          />
          <textarea
            value={body}
            onChange={(event) => setBody(event.target.value)}
            placeholder="Prompt text…"
            rows={4}
            spellCheck={false}
          />
          <div className="sub-form-actions">
            <button className="btn" onClick={() => setOpen(false)}>
              Cancel
            </button>
            <button
              className="btn primary"
              disabled={!body.trim()}
              onClick={() => {
                addPrompt(title, body);
                setTitle("");
                setBody("");
                setOpen(false);
              }}
            >
              Save
            </button>
          </div>
        </div>
      )}

      <div className="sub-group">Saved prompts</div>
      {prompts.length === 0 && <div className="sub-hint">None saved yet.</div>}

      <div className="prompt-grid">
        {prompts.map((prompt, index) => (
          <div key={prompt.id} className={removing.has(prompt.id) ? "prompt-gem is-leaving" : "prompt-gem"}>
            <button
              type="button"
              className="prompt-gem-main"
              onClick={() => setDraft(prompt.body)}
              title={prompt.body}
            >
              <span className={`prompt-gem-icon tone-${index % 4}`}>
                <Icon name="sparkles" size={16} />
              </span>
              <span className="prompt-gem-title">{prompt.title}</span>
              <span className="prompt-gem-body">{prompt.body}</span>
              <span className="prompt-gem-use">
                Use
                <Icon name="chevronRight" size={11} />
              </span>
            </button>
            <div className="prompt-gem-actions">
              <button
                className="sub-icon-btn"
                onClick={() => void navigator.clipboard.writeText(prompt.body)}
                title="Copy prompt"
                aria-label={`Copy ${prompt.title}`}
              >
                <Icon name="copy" size={11} />
              </button>
              <button
                className="sub-icon-btn"
                onClick={() => removeWithMotion(prompt.id)}
                title="Delete prompt"
                aria-label={`Delete ${prompt.title}`}
              >
                <Icon name="close" size={11} />
              </button>
            </div>
          </div>
        ))}
      </div>
    </>
  );
}

function MonitorView() {
  const status = useChat((s) => s.status);
  const [stats, setStats] = useState<SystemStats | null>(null);

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      const next = await window.dizel?.systemStats();
      if (alive && next) setStats(next);
    };
    void tick();
    const id = window.setInterval(tick, 2000);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, []);

  const info = status?.model_info ?? {};

  return (
    <>
      <div className="stat-card">
        <div className="stat-card-title">System Resources</div>
        <div className="stat-row">
          <span>CPU</span>
          <b>{stats ? `${stats.cpu} %` : "—"}</b>
        </div>
        <Meter value={stats?.cpu ?? 0} />
        <div className="stat-row">
          <span>RAM</span>
          <b>
            {stats ? `${stats.ram} % · ${stats.ramUsedMb} / ${stats.ramTotalMb} MB` : "—"}
          </b>
        </div>
        <Meter value={stats?.ram ?? 0} />
      </div>

      <div className="stat-card">
        <div className="stat-card-title">Current Session</div>
        <div className="stat-row">
          <span>Provider</span>
          <b>{status?.provider || "—"}</b>
        </div>
        <div className="stat-row">
          <span>Model</span>
          <b>{info.provider || "—"}</b>
        </div>
        <div className="stat-row">
          <span>Params</span>
          <b>{info.params || "—"}</b>
        </div>
        <div className="stat-row">
          <span>Context</span>
          <b>{info.ctx_len ? `${info.ctx_len} tokens` : "—"}</b>
        </div>
        <div className="stat-row">
          <span>Device</span>
          <b>{info.device ? String(info.device).toUpperCase() : "—"}</b>
        </div>
      </div>
    </>
  );
}

export function SecondarySidebar() {
  const view = useChat((s) => s.workspaceView);
  const close = useChat((s) => s.closeWorkspace);

  if (!view) return null;

  return (
    <aside className="subsidebar">
      <header className="sub-head">
        <h2>{TITLES[view]}</h2>
        <button className="sub-icon-btn" onClick={close} title="Close panel">
          <Icon name="close" size={13} />
        </button>
      </header>

      <div className="sub-body" key={view}>
        {view === "Chats" && <ChatsView />}
        {view === "Riset" && <RisetView />}
        {view === "Library" && <LibraryView />}
        {view === "Monitor" && <MonitorView />}

        {view === "Image" && (
          <Empty
            icon="image"
            title="Image gallery"
            body="Images you attach to a message will be collected here."
            hint="Attachments are text-only for now — this fills in once vision lands."
          />
        )}
        {view === "Archived" && <ArchivedView />}
        {view === "Presentation" && (
          <Empty
            icon="file"
            title="Presentations"
            body="Slide decks you generate will be listed here."
            hint="Not implemented yet — the desktop app doesn't generate slides either."
          />
        )}
      </div>
    </aside>
  );
}
