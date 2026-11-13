import { useEffect, useRef, useState } from "react";

import { ChatView } from "./components/ChatView";
import { PluginsDialog } from "./components/PluginsDialog";
import { ToolsSheet } from "./components/ToolsSheet";
import { Composer } from "./components/Composer";
import { AttachmentPreview } from "./components/AttachmentPreview";
import { exportName, transcriptToMarkdown } from "./lib/transcript";
import { Icon } from "./components/Icon";
import { ProfileDialog } from "./components/ProfileDialog";
import { SecondarySidebar } from "./components/SecondarySidebar";
import { Sidebar } from "./components/Sidebar";
import { SettingsDialog } from "./components/SettingsDialog";
import { StatusBar } from "./components/StatusBar";
import { DebugPanel, Toast } from "./components/DebugPanel";
import { TitleBar } from "./components/TitleBar";
import { useChat } from "./store/chatStore";
import { useUpdates } from "./hooks/useUpdates";
import { contextUsage } from "./services/context";
import { modelName, providerName } from "./services/model";
import type { ChatStatus } from "./types";

function headerState(
  status: ChatStatus | null,
  booted: boolean,
): { text: string; loading: boolean; failed: boolean } {
  if (!booted) {
    return { text: "Starting Dizel…", loading: true, failed: false };
  }

  const load = status?.load;
  const provider = providerName(status);
  const remote = Boolean(status?.provider && status.provider !== "local");

  if (load?.status === "loading") {
    return {
      text: remote ? `${provider} · connecting…` : load.message || "Loading model…",
      loading: true,
      failed: false,
    };
  }
  if (load?.status === "error") {
    return {
      text: remote ? `${provider} · connection failed` : "Model failed to load",
      loading: false,
      failed: true,
    };
  }
  if (status?.is_ready) {
    if (remote) {
      const bits = [provider];
      if (status.api_model) bits.push(status.api_model);
      return { text: bits.join(" · "), loading: false, failed: false };
    }
    const info = status.model_info;
    const bits = [modelName(status)];
    if (info.params) bits.push(info.params);
    if (info.d_model) {
      bits.push(info.n_layers ? `d=${info.d_model} L=${info.n_layers}` : `d=${info.d_model}`);
    }
    if (info.device) bits.push(String(info.device).toUpperCase());
    return { text: bits.join(" · "), loading: false, failed: false };
  }
  return {
    text: remote ? `${provider} · no model selected` : "No model loaded",
    loading: false,
    failed: false,
  };
}

export default function App() {
  const boot = useChat((s) => s.boot);
  const bootError = useChat((s) => s.bootError);
  const error = useChat((s) => s.error);
  const dismissError = useChat((s) => s.dismissError);
  const status = useChat((s) => s.status);
  const streaming = useChat((s) => s.streaming);
  const theme = useChat((s) => s.theme);
  const openSettings = useChat((s) => s.openSettings);
  const statusBarOpen = useChat((s) => s.statusBarOpen);
  const toggleStatusBar = useChat((s) => s.toggleStatusBar);
  const booted = useChat((s) => s.booted);
  const newChat = useChat((s) => s.newChat);
  const toggleSidebar = useChat((s) => s.toggleSidebar);
  const attachPaths = useChat((s) => s.attachPaths);
  const handoff = useChat((s) => s.handoff);
  const debug = useChat((s) => s.debug);
  const messages = useChat((s) => s.messages);
  const updateState = useUpdates();
  const updateNotified = useRef("");

  useEffect(() => {
    if (!updateState || updateState.status !== "downloaded" || !updateState.latestVersion) {
      return;
    }
    if (updateNotified.current === updateState.latestVersion) return;
    updateNotified.current = updateState.latestVersion;
    useChat
      .getState()
      .notify(`Dizel ${updateState.latestVersion} is ready — restart to install`);
  }, [updateState]);

  // The generating row keeps animating for `holdMs` after the reply has already
  // arrived, so `streaming` alone turns the glow off mid-animation. Tick while a
  // recent message is still inside its hold window.
  const [nowTick, setNowTick] = useState(Date.now());
  const tail = messages[messages.length - 1];
  const holdingTail =
    tail?.role === "assistant" && Boolean(tail.holdMs)
      ? nowTick - tail.createdAt < (tail.holdMs ?? 0)
      : false;

  useEffect(() => {
    if (!holdingTail) return undefined;
    const id = window.setInterval(() => setNowTick(Date.now()), 200);
    return () => window.clearInterval(id);
  }, [holdingTail]);
  const sessionId = useChat((s) => s.sessionId);
  const [dragging, setDragging] = useState(false);
  const [ctxWarnDismissed, setCtxWarnDismissed] = useState(false);

  const budgetOverride = useChat((s) => s.budgetOverride);
  const serverUsage = useChat((s) => s.serverUsage);
  const sidecarDown = useChat((s) => s.sidecarDown);
  const restartSidecar = useChat((s) => s.restartSidecar);
  const localUsage = contextUsage(messages, status?.model_info?.ctx_len);
  // The server counts what the model actually receives. The local estimate also
  // counts messages that were never sent — after /ctxfull it claimed 100% while
  // the server reported 127 of 2,048.
  const baseUsage = serverUsage
    ? { used: serverUsage.used, total: serverUsage.total, percent: serverUsage.percent }
    : localUsage;
  // Debug /budget forces the displayed numbers so the warning can be exercised.
  const usage =
    budgetOverride === null
      ? baseUsage
      : {
          used: budgetOverride,
          total: baseUsage.total,
          percent: Math.min(100, Math.round((budgetOverride * 100) / baseUsage.total)),
        };
  // Re-arm the warning whenever the conversation changes.
  useEffect(() => setCtxWarnDismissed(false), [sessionId]);
  const showCtxWarning =
    (messages.length > 0 || budgetOverride !== null) && usage.percent >= 80 && !ctxWarnDismissed;

  useEffect(() => {
    void boot();
  }, [boot]);

  // The model process can die at any moment — a bad checkpoint, an OOM, a tool
  // that took the interpreter with it. Without this the UI just stops answering.
  useEffect(() => {
    const off = window.dizel?.onSidecarDown?.((info) => {
      useChat.setState({ sidecarDown: info });
    });
    return () => off?.();
  }, []);

  // The tray's "New chat" entry talks to the renderer over IPC.
  useEffect(() => {
    window.dizel?.onNewChat?.(() => newChat());
  }, [newChat]);

  // Global shortcuts. Listed in Settings → Shortcuts.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (!(event.ctrlKey || event.metaKey)) return;
      const key = event.key.toLowerCase();
      const state = useChat.getState();

      if (event.shiftKey && key === "c" && state.prefs.copyLastReplyHotkey) {
        event.preventDefault();
        const lastReply = [...state.messages].reverse().find((m) => m.role === "assistant");
        if (!lastReply?.content) {
          state.notify("No reply to copy yet.");
        } else {
          navigator.clipboard.writeText(lastReply.content).then(
            () => state.notify(`Copied ${lastReply.content.length} characters.`),
            () => state.notify("Clipboard unavailable."),
          );
        }
      } else if (event.shiftKey && key === "x" && state.prefs.stopGenerationHotkey) {
        event.preventDefault();
        if (state.streaming) {
          void state.stop();
          state.notify("Stopped generation.");
        }
      } else if (key === "n") {
        event.preventDefault();
        newChat();
      } else if (key === "k") {
        event.preventDefault();
        document.querySelector<HTMLTextAreaElement>(".composer textarea")?.focus();
      } else if (key === "b") {
        event.preventDefault();
        toggleSidebar();
      } else if (event.key === ",") {
        event.preventDefault();
        openSettings();
      }
    };

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [newChat, toggleSidebar, openSettings]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  // Drag files anywhere onto the window to attach them.
  useEffect(() => {
    const hasFiles = (event: DragEvent) => Boolean(event.dataTransfer?.types?.includes("Files"));

    const onOver = (event: DragEvent) => {
      if (!hasFiles(event)) return;
      event.preventDefault();
      setDragging(true);
    };

    // dragleave also fires when moving between children; only a null
    // relatedTarget means the pointer actually left the window.
    const onLeave = (event: DragEvent) => {
      if (event.relatedTarget) return;
      setDragging(false);
    };

    const onDrop = (event: DragEvent) => {
      event.preventDefault();
      setDragging(false);
      const paths = Array.from(event.dataTransfer?.files ?? [])
        .map((file) => window.dizel?.pathForFile(file) ?? "")
        .filter(Boolean);
      if (paths.length > 0) void attachPaths(paths);
    };

    window.addEventListener("dragover", onOver);
    window.addEventListener("dragleave", onLeave);
    window.addEventListener("drop", onDrop);
    return () => {
      window.removeEventListener("dragover", onOver);
      window.removeEventListener("dragleave", onLeave);
      window.removeEventListener("drop", onDrop);
    };
  }, [attachPaths]);

  const header = headerState(status, booted);
  const modelLoading = status?.load?.status === "loading";
  // On an empty, ready chat the composer sits inside the welcome flow; once
  // there's a conversation it drops back to its dock at the bottom.
  // Any empty chat keeps the composer inside the welcome flow — including while
  // the model loads, so it doesn't jump between layouts during startup.
  const inlineComposer = messages.length === 0;
  const pillClass = [
    "status-pill",
    header.loading ? "is-loading" : "",
    header.failed ? "is-failed" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className="app">
      <TitleBar />
      <ToolsSheet />

      <div className="app-body">
        <Sidebar />
        <SecondarySidebar />

        <main
          className={
            streaming || holdingTail || messages.length === 0 ? "main is-glowing" : "main"
          }
        >
          {/* The split starts here, not at ChatView: the file panel has to reach
              the top of the window, past the header and the load bar. */}
          <div className="chat-split">
            <div className="chat-col">
              <div className="chat-header">
            <span className={pillClass} title={header.text}>
              {header.loading && <span className="pill-spinner" />}
              <span className="pill-text" key={header.text}>
                {header.text}
              </span>
            </span>

            {debug && (
              <span className="debug-badge" title="Debug mode — type /help in the composer">
                DEBUG
              </span>
            )}

            <div className="header-actions">
              {messages.length > 0 && (
              <button
                className="hdr-btn"
                onClick={() => {
                  const firstUser = messages.find((m) => m.role === "user");
                  const title = (firstUser?.content ?? "").trim().split("\n")[0].slice(0, 60);
                  void window.dizel?.saveText({
                    defaultName: exportName(),
                    contents: transcriptToMarkdown(messages, title),
                  });
                }}
                title="Export this conversation as Markdown"
                aria-label="Export conversation"
              >
                <Icon name="download" size={14} />
              </button>
              )}
              <button
                className="hdr-btn"
                onClick={toggleStatusBar}
                title={statusBarOpen ? "Hide status bar" : "Show status bar"}
                aria-label={statusBarOpen ? "Hide status bar" : "Show status bar"}
              >
                <Icon name={statusBarOpen ? "chevronDown" : "chevronUp"} size={14} />
              </button>
              <button
                className="hdr-btn"
                onClick={() => openSettings()}
                title="Settings"
                aria-label="Settings"
              >
                <Icon name="settings" size={15} />
              </button>
            </div>
          </div>

          <div
            className={modelLoading || streaming ? "loadbar is-loading" : "loadbar"}
            aria-hidden="true"
          >
            <span />
          </div>

              <ChatView />

          {showCtxWarning && (
            <div className="banner-wrap">
              <div className="banner warn">
                <Icon name="alert" size={13} style={{ flexShrink: 0, marginTop: 2 }} />
                <span style={{ flex: 1 }}>
                  {usage.percent}% of the {usage.total}-token context is used — older
                  messages will start dropping from this conversation.
                </span>
                <button className="banner-action" onClick={() => void handoff()}>
                  Continue in a new chat
                </button>
                <button
                  className="banner-action"
                  onClick={() => {
                    newChat();
                    setCtxWarnDismissed(false);
                  }}
                  title="Clear this conversation and start with an empty context"
                >
                  Reset context
                </button>
                <button
                  className="banner-x"
                  onClick={() => setCtxWarnDismissed(true)}
                  aria-label="Dismiss"
                >
                  <Icon name="close" size={12} />
                </button>
              </div>
            </div>
          )}

          {sidecarDown && (
            <div className="banner-wrap">
              <div className="banner">
                <Icon name="alert" size={13} style={{ flexShrink: 0, marginTop: 2 }} />
                <span style={{ flex: 1 }}>
                  The model process stopped
                  {sidecarDown.code !== null ? ` (exit ${sidecarDown.code})` : ""}. Nothing
                  will answer until it comes back.
                </span>
                <button className="banner-action" onClick={() => void restartSidecar()}>
                  Restart
                </button>
              </div>
            </div>
          )}

          {(bootError || error) && (
            <div className="banner-wrap">
              <div className="banner">
                <Icon name="close" size={13} style={{ flexShrink: 0, marginTop: 2 }} />
                <span style={{ flex: 1 }}>{bootError ?? error}</span>
                <button className="banner-x" onClick={dismissError} aria-label="Dismiss">
                  <Icon name="close" size={12} />
                </button>
              </div>
            </div>
          )}

              {!inlineComposer && <Composer />}
            </div>

            <AttachmentPreview />
          </div>
          <div className={statusBarOpen ? "statusbar-wrap is-open" : "statusbar-wrap"}>
            <StatusBar />
          </div>
          <DebugPanel />
          <Toast />
        </main>
      </div>

      <SettingsDialog />
      <ProfileDialog />
      <PluginsDialog />

      {dragging && (
        <div className="drop-overlay">
          <div className="drop-card">
            <Icon name="file" size={22} />
            <strong>Drop to attach</strong>
            <span>Text and code files, up to 4 per message</span>
          </div>
        </div>
      )}
    </div>
  );
}
