import {
  startTransition,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { Icon } from "./Icon";
import { MaskIcon } from "./MaskIcon";
import type { MaskIconName } from "./MaskIcon";
import { Picker } from "./Picker";
import { SessionRow } from "./SessionRow";
import { Skeleton } from "./Skeleton";
import { useChat } from "../store/chatStore";
import { useUpdates } from "../hooks/useUpdates";
import type { WorkspaceView } from "../types";

/** Recent chats page in twenties; after forty, search is the way further back. */
const INITIAL_LIMIT = 20;
const PAGE_SIZE = 20;
const MAX_LIMIT = 40;

const WORKSPACES: { view: WorkspaceView; label: string; icon: MaskIconName }[] = [
  { view: "Image", label: "Image", icon: "image" },
  { view: "Presentation", label: "Presentation", icon: "layout" },
  { view: "Riset", label: "Riset", icon: "search" },
  { view: "Archived", label: "Archived", icon: "archive" },
  { view: "Library", label: "Library", icon: "book" },
  { view: "Monitor", label: "Monitor", icon: "activity" },
];

export function Sidebar() {
  const collapsed = useChat((s) => s.sidebarCollapsed);
  const toggleSidebar = useChat((s) => s.toggleSidebar);
  const newChat = useChat((s) => s.newChat);
  const sessions = useChat((s) => s.sessions);
  const sessionId = useChat((s) => s.sessionId);
  const username = useChat((s) => s.username);
  const profile = useChat((s) => s.profile);
  const showRecent = useChat((s) => s.showRecent);
  const toggleRecent = useChat((s) => s.toggleRecent);
  const workspaceView = useChat((s) => s.workspaceView);
  const openWorkspace = useChat((s) => s.openWorkspace);
  const avatarUrl = useChat((s) => s.avatarUrl);
  const openProfile = useChat((s) => s.openProfile);
  const openSettings = useChat((s) => s.openSettings);
  const booted = useChat((s) => s.booted);
  const update = useUpdates();

  const [menuOpen, setMenuOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [visibleLimit, setVisibleLimit] = useState(INITIAL_LIMIT);
  const asideRef = useRef<HTMLElement>(null);
  const firstWidthChange = useRef(true);

  // The animated ambient glow is a full-size pseudo-element, so resizing the
  // main pane repaints it every frame and dominates the animation cost. Swap
  // it for a static, viewport-sized fallback only while the width transitions.
  useLayoutEffect(() => {
    if (firstWidthChange.current) {
      firstWidthChange.current = false;
      return;
    }
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const aside = asideRef.current;
    if (!aside) return;
    aside.classList.add("is-resizing");
    const timer = window.setTimeout(() => aside.classList.remove("is-resizing"), 190);
    return () => {
      window.clearTimeout(timer);
      aside.classList.remove("is-resizing");
    };
  }, [collapsed]);
  // Keep the history mounted after the first open so collapsing never unmounts
  // and rebuilding a long row list. The list also keeps a fixed width while the
  // shell animates, so each width frame clips it instead of relaying out rows.
  const [historyMounted, setHistoryMounted] = useState(!collapsed);

  useEffect(() => {
    if (!collapsed && !historyMounted) {
      startTransition(() => setHistoryMounted(true));
    }
  }, [collapsed, historyMounted]);

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const live = sessions.filter((session) => !session.archived);
    if (!needle) return live;
    return live.filter((session) =>
      `${session.title} ${session.preview}`.toLowerCase().includes(needle),
    );
  }, [sessions, query]);

  const pinned = visible.filter((session) => session.pinned);
  const rest = visible.filter((session) => !session.pinned);
  const searching = query.trim().length > 0;
  // Search looks through everything; the recents list is what pages.
  const shownRest = searching ? rest : rest.slice(0, visibleLimit);
  const hiddenCount = rest.length - shownRest.length;
  const canShowMore = !searching && hiddenCount > 0 && visibleLimit < MAX_LIMIT;
  const reachedSearchLimit = !searching && hiddenCount > 0 && visibleLimit >= MAX_LIMIT;

  return (
    <aside ref={asideRef} className={collapsed ? "sidebar collapsed" : "sidebar"}>
      <div className="sb-brand">
        <span className="sb-brand-text">
          <span className="sb-title">Dizel</span>
        </span>
        <button
          className="sb-collapse"
          onClick={toggleSidebar}
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          <Icon name={collapsed ? "panelExpand" : "panelCollapse"} size={20} />
        </button>
      </div>

      <nav className="sb-nav">
        <button className="nav-item" onClick={newChat} title="New chat">
          <MaskIcon name="edit-3" size={20} />
          <span className="nav-label">New chat</span>
        </button>

        <button
          className={workspaceView === "Chats" ? "nav-item is-active" : "nav-item"}
          onClick={() => openWorkspace("Chats")}
          title="Chats"
        >
          <MaskIcon name="new-chat-2" size={22} />
          <span className="nav-label">Chats</span>
        </button>

        <Picker
          open={menuOpen}
          onOpenChange={setMenuOpen}
          triggerClassName={
            workspaceView && workspaceView !== "Chats" ? "nav-item is-active" : "nav-item"
          }
          triggerTitle="Workspaces"
          placement="down"
          label={
            <>
              <MaskIcon name="workspace-2" size={20} />
              <span className="nav-label">Workspaces</span>
            </>
          }
        >
          {WORKSPACES.map((item) => (
            <button
              key={item.view}
              className={
                item.view === workspaceView
                  ? "popover-item with-icon active"
                  : "popover-item with-icon"
              }
              onClick={() => {
                setMenuOpen(false);
                openWorkspace(item.view);
              }}
            >
              <MaskIcon name={item.icon} size={18} />
              <span className="popover-item-text">{item.label}</span>
              {item.view === workspaceView && <Icon name="check" size={13} />}
            </button>
          ))}
        </Picker>
      </nav>

      {historyMounted && (
        <div className={collapsed ? "sb-history-clip is-hidden" : "sb-history-clip"}>
          <div className="sb-history">
            <div className="sb-divider" />
            <div className="sb-label">
              <span>Recent Chats</span>
              <button
                className="sb-label-btn"
                onClick={toggleRecent}
                title={showRecent ? "Hide recent chats" : "Show recent chats"}
                aria-label={showRecent ? "Hide recent chats" : "Show recent chats"}
              >
                <Icon name={showRecent ? "chevronUp" : "chevronDown"} size={12} />
              </button>
            </div>
            <div className={showRecent ? "sb-recent is-open" : "sb-recent"}>
              <div className="sb-recent-inner">
                {booted && sessions.length > 6 && (
                  <div className="sb-search">
                    <Icon name="search" size={12} />
                    <input
                      value={query}
                      onChange={(event) => setQuery(event.target.value)}
                      placeholder="Search chats…"
                      spellCheck={false}
                    />
                  </div>
                )}

                <div className="sb-list">
                  {!booted &&
                    Array.from({ length: 7 }).map((_, index) => (
                      <div className="sb-row-skel" key={index}>
                        <Skeleton width={`${52 + ((index * 17) % 40)}%`} height={9} />
                      </div>
                    ))}

                  {booted && pinned.length > 0 && <div className="sb-sub">Pinned</div>}
                  {booted &&
                    pinned.map((session) => (
                      <SessionRow
                        key={session.id}
                        session={session}
                        active={session.id === sessionId}
                      />
                    ))}

                  {booted && pinned.length > 0 && shownRest.length > 0 && (
                    <div className="sb-sub">All chats</div>
                  )}
                  {booted &&
                    shownRest.map((session) => (
                      <SessionRow
                        key={session.id}
                        session={session}
                        active={session.id === sessionId}
                      />
                    ))}

                  {booted && canShowMore && (
                    <button
                      className="sb-more"
                      onClick={() =>
                        setVisibleLimit((limit) => Math.min(MAX_LIMIT, limit + PAGE_SIZE))
                      }
                      aria-label={`Show ${Math.min(PAGE_SIZE, hiddenCount)} more chats`}
                    >
                      Show more
                      <Icon name="chevronDown" size={12} />
                    </button>
                  )}

                  {booted && reachedSearchLimit && (
                    <div className="sb-search-hint">
                      <Icon name="search" size={12} />
                      <span>Showing {MAX_LIMIT} most recent. Use search for older chats.</span>
                    </div>
                  )}

                  {booted && visible.length === 0 && (
                    <div className="sb-empty">{query.trim() ? "No matches" : "No chats yet"}</div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {(!showRecent || !historyMounted) && <div className="sidebar-spacer" />}

      {update &&
        (update.status === "available" ||
          update.status === "downloading" ||
          update.status === "downloaded") && (
          <button
            className={collapsed ? "sb-update is-collapsed" : "sb-update"}
            onClick={() => openSettings("About")}
            title={
              update.status === "downloaded"
                ? `Dizel ${update.latestVersion} is ready — open Settings to restart`
                : update.status === "downloading"
                  ? `Downloading Dizel ${update.latestVersion} — ${update.progress}%`
                  : `Dizel ${update.latestVersion} is available`
            }
            aria-label="Open update settings"
          >
            <span className="sb-update-icon">
              <Icon name="download" size={14} />
            </span>
            {!collapsed && (
              <>
                <span className="sb-update-text">
                  {update.status === "downloaded"
                    ? "Restart to update"
                    : update.status === "downloading"
                      ? "Downloading update"
                      : "Update available"}
                </span>
                <span className="sb-update-meta">
                  {update.status === "downloading"
                    ? `${update.progress}%`
                    : `v${update.latestVersion}`}
                </span>
              </>
            )}
          </button>
        )}

      <button
        className="sb-foot"
        onClick={openProfile}
        title={
          [username, profile.pronouns, profile.bio].filter(Boolean).join(" · ") || "Edit profile"
        }
      >
        <span className="sb-avatar-ring" data-accent={profile.accent}>
          <span className="sb-avatar" data-accent={profile.accent}>
            {avatarUrl ? <img src={avatarUrl} alt="" /> : username.slice(0, 1).toUpperCase()}
          </span>
        </span>
        <span className="sb-user">
          {username}
          {profile.pronouns && <small className="sb-user-pronouns">{profile.pronouns}</small>}
        </span>
      </button>
    </aside>
  );
}
