import { useEffect, useMemo, useRef, useState } from "react";

import { CommandDrawer } from "./CommandDrawer";
import { Icon } from "./Icon";
import { kindOf } from "./AttachmentPreview";
import { ModelIcon } from "./Logo";
import { ProviderAvatar } from "./ProviderAvatar";
import { Skeleton } from "./Skeleton";
import { modelName } from "../services/model";
import { Picker } from "./Picker";
import { VerbosityPicker } from "./VerbosityPicker";
import { ModePicker } from "./ModePicker";
import { useChat } from "../store/chatStore";
import type { ScorecardSummary } from "../types";
import {
  buildCommandGroups,
  filterCommands,
  flattenGroups,
  type CommandEntry,
  type CommandGroup,
} from "../services/commands";

/** Response modes for local models; profiles supply the per-family lists. */
const DEFAULT_MODES = ["Fast", "Planning", "Coding", "Thinking"];

/** The reply-length ladder. Server-side it scales the reply's token budget. */
const extension = (name: string) => {
  const parts = name.split(".");
  return parts.length > 1 ? (parts.pop() as string).slice(0, 3).toUpperCase() : "TXT";
};

const fileName = (path: string) => path.split(/[\\/]/).pop() || path;

const formatSize = (bytes: number) => {
  if (bytes >= 1048576) return `${(bytes / 1048576).toFixed(1)} MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${bytes} B`;
};

type LocalEntry = {
  path: string;
  display_name: string;
  kind: string;
  size: number;
  exists?: boolean;
  scorecard?: ScorecardSummary | null;
};

export function Composer() {
  const draft = useChat((s) => s.draft);
  const setDraft = useChat((s) => s.setDraft);
  const send = useChat((s) => s.send);
  const stop = useChat((s) => s.stop);
  const streaming = useChat((s) => s.streaming);
  const ready = useChat((s) => s.status?.is_ready ?? false);

  const attachments = useChat((s) => s.attachments);
  const attachedSkills = useChat((s) => s.attachedSkills);
  const detachSkill = useChat((s) => s.detachSkill);
  const recentPaths = useChat((s) => s.recentPaths);
  const prompts = useChat((s) => s.prompts);
  const addFiles = useChat((s) => s.addFiles);
  const addFolder = useChat((s) => s.addFolder);
  const addImages = useChat((s) => s.addImages);
  const attachPaths = useChat((s) => s.attachPaths);
  const removeAttachment = useChat((s) => s.removeAttachment);
  const visionReady = useChat((s) => s.status?.model_info?.vision === true);

  // Empty state: the composer sits mid-screen, so the menu drops down into the
  // space below it. In a conversation the composer is pinned to the bottom and a
  // downward menu would fall off the window, so it opens upward.
  const emptyState = useChat((s) => s.messages.length === 0);
  const placement = emptyState ? "down" : "up";
  const mode = useChat((s) => s.mode);
  const setMode = useChat((s) => s.setMode);
  const thinkingEffort = useChat((s) => s.thinkingEffort);
  const setThinkingEffort = useChat((s) => s.setThinkingEffort);
  const fastSpeed = useChat((s) => s.fastSpeed);
  const setFastSpeed = useChat((s) => s.setFastSpeed);
  const profiles = useChat((s) => s.profiles);

  const micState = useChat((s) => s.micState);
  const micError = useChat((s) => s.micError);
  const micAction = useChat((s) => s.micAction);
  const dictation = useChat((s) => s.dictation);
  const startDictation = useChat((s) => s.startDictation);
  const cancelDictation = useChat((s) => s.cancelDictation);
  const dismissDictation = useChat((s) => s.dismissDictation);
  const debug = useChat((s) => s.debug);
  const runSlash = useChat((s) => s.runSlash);
  const skills = useChat((s) => s.skills);
  const booted = useChat((s) => s.booted);
  const enterToSend = useChat((s) => s.prefs.enterToSend);
  const spellcheck = useChat((s) => s.prefs.spellcheck);
  const autoFocusComposer = useChat((s) => s.prefs.autoFocusComposer);
  const escapeClearsDraft = useChat((s) => s.prefs.escapeClearsDraft);
  const arrowUpEditsLast = useChat((s) => s.prefs.arrowUpEditsLast);
  const sessionId = useChat((s) => s.sessionId);

  const [modeOpen, setModeOpen] = useState(false);
  const [modelOpen, setModelOpen] = useState(false);
  const [allModelsOpen, setAllModelsOpen] = useState(false);
  const [toolsOpen, setToolsOpen] = useState(false);
  // The chip and the side panel share one selection, so the panel highlights the
  // file whose chip is open.
  const openPreview = useChat((s) => s.preview);
  const setOpenPreview = useChat((s) => s.setPreview);

  const status = useChat((s) => s.status);
  const providers = useChat((s) => s.providers);
  const providerModels = useChat((s) => s.providerModels);
  const providerModelsError = useChat((s) => s.providerModelsError);
  const selectModel = useChat((s) => s.selectModel);
  const loadProviderModels = useChat((s) => s.loadProviderModels);
  const recentModels = useChat((s) => s.recentModels);
  const localModels = useChat((s) => s.models);
  const loadRecentModels = useChat((s) => s.loadRecentModels);
  const switchModel = useChat((s) => s.switchModel);
  const openSettings = useChat((s) => s.openSettings);
  const editingId = useChat((s) => s.editingId);
  const cancelEdit = useChat((s) => s.cancelEdit);

  const activeSlug = status?.provider ?? "local";
  const isLocal = activeSlug === "local";
  const activeProvider = providers.find((provider) => provider.slug === activeSlug);

  const localName = modelName(status);
  const localMeta =
    [status?.model_info?.params, status?.model_info?.device?.toString().toUpperCase()]
      .filter(Boolean)
      .join(" · ") || "Not loaded";
  const localLower = localName.toLowerCase();
  const modelFamily = isLocal && ready
    ? /^dizel\b/.test(localLower)
      ? "Dizel"
      : /^mila\b/.test(localLower)
        ? "Mila"
        : "Generic"
    : "";

  const modelLabel = isLocal
    ? localName
    : status?.api_model || activeProvider?.name || activeSlug;

  const loadedPath = status?.load?.checkpoint ?? "";
  const normalizedPath = (path: string) => path.replace(/\\/g, "/").toLowerCase();
  const samePath = (a: string, b: string) => {
    const left = normalizedPath(a);
    const right = normalizedPath(b);
    return (
      Boolean(left && right) &&
      (left === right || left.endsWith(`/${right}`) || right.endsWith(`/${left}`))
    );
  };
  const recentKeys = new Set(recentModels.map((entry) => normalizedPath(entry.path)));
  const otherModels = localModels.filter(
    (entry) => !recentKeys.has(normalizedPath(entry.path)),
  );

  const pickLocal = async (path: string) => {
    if (!isLocal) await selectModel("local", "");
    await switchModel(path);
  };

  const renderLocalEntry = (entry: LocalEntry, active: boolean) => (
    <button
      key={entry.path}
      className={`model-row${active ? " active" : ""}${entry.exists ? "" : " is-missing"}`}
      disabled={!entry.exists}
      title={entry.exists ? entry.path : "This checkpoint file is no longer on disk."}
      onClick={() => {
        setModelOpen(false);
        void pickLocal(entry.path);
      }}
    >
      <span className="model-row-icon">
        <ModelIcon variant="dizel" size={22} />
      </span>
      <span className="model-row-text">
        <b>{entry.display_name}</b>
        <small>
          {entry.exists
            ? `${formatSize(entry.size)} · ${entry.kind.toUpperCase()}`
            : "Missing file"}
        </small>
      </span>
      {entry.scorecard ? (
        <span
          className="model-score"
          title={
            `${entry.scorecard.passed}/${entry.scorecard.total} core cases · ` +
            Object.entries(entry.scorecard.by_category ?? {})
              .map(([name, score]) => `${name} ${Number(score).toFixed(2)}`)
              .join(" · ") +
            ` · ${entry.scorecard.evaluated_at}`
          }
        >
          {entry.scorecard.passed}/{entry.scorecard.total}
        </span>
      ) : null}
      {active && <Icon name="check" size={13} />}
    </button>
  );



  const ref = useRef<HTMLTextAreaElement>(null);
  const composerRef = useRef<HTMLDivElement>(null);
  const drawerHostRef = useRef<HTMLDivElement>(null);

  // A bare slash prefix is a command picker; once the first argument starts,
  // the drawer gets out of the way.
  const slash = /^\/(\S*)$/.exec(draft);
  const slashQuery = slash ? slash[1].toLowerCase() : null;
  const commandGroups = useMemo(
    () => (slashQuery === null ? [] : filterCommands(buildCommandGroups(skills, debug), slashQuery)),
    [skills, debug, slashQuery],
  );
  const commandMatches = useMemo(() => flattenGroups(commandGroups), [commandGroups]);
  const [drawerDismissed, setDrawerDismissed] = useState(false);
  const [activeCommand, setActiveCommand] = useState(0);
  const drawerOpen =
    slashQuery !== null && !drawerDismissed && commandMatches.length > 0 && !editingId;
  const activeCommandIndex = Math.min(activeCommand, Math.max(0, commandMatches.length - 1));

  // Keep the tray mounted through its exit animation, with its last content.
  const [drawerMounted, setDrawerMounted] = useState(false);
  const [drawerClosing, setDrawerClosing] = useState(false);
  const drawerCache = useRef<{
    groups: CommandGroup[];
    count: number;
    query: string;
    activeIndex: number;
  }>({ groups: [], count: 0, query: "", activeIndex: 0 });
  const closeTimer = useRef<number | null>(null);

  useEffect(() => {
    if (!drawerOpen) return;
    drawerCache.current = {
      groups: commandGroups,
      count: commandMatches.length,
      query: slashQuery ?? "",
      activeIndex: activeCommandIndex,
    };
  }, [drawerOpen, commandGroups, commandMatches, slashQuery, activeCommandIndex]);

  useEffect(() => {
    if (drawerOpen) {
      if (closeTimer.current !== null) {
        window.clearTimeout(closeTimer.current);
        closeTimer.current = null;
      }
      setDrawerClosing(false);
      setDrawerMounted(true);
      return;
    }
    if (!drawerMounted) return;

    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    setDrawerClosing(true);
    closeTimer.current = window.setTimeout(
      () => {
        setDrawerMounted(false);
        setDrawerClosing(false);
        closeTimer.current = null;
      },
      reduce ? 0 : 150,
    );
    return () => {
      if (closeTimer.current !== null) {
        window.clearTimeout(closeTimer.current);
        closeTimer.current = null;
      }
    };
  }, [drawerOpen, drawerMounted]);

  useEffect(() => {
    setDrawerDismissed(false);
    setActiveCommand(0);
  }, [slashQuery, debug]);

  useEffect(() => {
    if (toolsOpen || modelOpen || modeOpen) setDrawerDismissed(true);
  }, [toolsOpen, modelOpen, modeOpen]);

  useEffect(() => {
    if (modelOpen) void loadRecentModels();
  }, [modelOpen, loadRecentModels]);

  useEffect(() => {
    if (!drawerOpen) return;
    const onPointerDown = (event: PointerEvent) => {
      if (!(event.target instanceof Node)) return;
      if (composerRef.current?.contains(event.target)) return;
      if (drawerHostRef.current?.contains(event.target)) return;
      setDrawerDismissed(true);
    };
    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, [drawerOpen]);

  useEffect(() => {
    if (!autoFocusComposer || !booted || !ready) return;
    window.requestAnimationFrame(() => ref.current?.focus());
  }, [autoFocusComposer, booted, ready, sessionId]);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [draft]);

  // Skills insert a long template; the user's own text sits at the end, so
  // focus the field and scroll there rather than leaving it above the fold.
  const composerFocus = useChat((s) => s.composerFocus);
  useEffect(() => {
    if (!composerFocus) return;
    window.requestAnimationFrame(() => {
      const el = ref.current;
      if (!el) return;
      el.focus();
      el.setSelectionRange(el.value.length, el.value.length);
      el.scrollTop = el.scrollHeight;
    });
  }, [composerFocus]);

  const modes = modelFamily && profiles[modelFamily]?.length
    ? profiles[modelFamily]
    : DEFAULT_MODES;

  // The loaded model decides the profile family; loading another family
  // re-applies the current mode so its prompt and sampling land on it.
  const appliedFamily = useRef("");
  useEffect(() => {
    if (!isLocal || !ready || !modelFamily) return;
    if (appliedFamily.current === modelFamily) return;
    appliedFamily.current = modelFamily;
    void setMode(mode);
  }, [isLocal, ready, modelFamily, mode, setMode]);

  const pasteText = async () => {
    let text = "";
    try {
      text = (await window.dizel?.readClipboardText?.()) ?? "";
      if (!text) text = await navigator.clipboard.readText();
    } catch {
      /* clipboard permission or empty clipboard */
    }
    const clean = text.replace(/\r\n/g, "\n").trim();
    if (!clean) {
      useChat.setState({ error: "Clipboard has no text to paste." });
      return;
    }
    const next = draft.trim() ? `${draft.replace(/\s+$/, "")}\n\n${clean}` : clean;
    setDraft(next);
    window.requestAnimationFrame(() => ref.current?.focus());
  };

  const insertCodeBlock = () => {
    const block = "```\n\n```";
    const next = draft.trim() ? `${draft.replace(/\s+$/, "")}\n\n${block}` : block;
    setDraft(next);
    window.requestAnimationFrame(() => {
      const field = ref.current;
      if (!field) return;
      field.focus();
      const caret = next.length - 3;
      field.setSelectionRange(caret, caret);
    });
  };

  const clearAttachments = () => {
    for (const file of attachments) removeAttachment(file.id);
    useChat.getState().notify?.("Attachments cleared");
  };

  const pickCommand = (entry: CommandEntry) => {
    setDrawerDismissed(true);
    if (entry.kind === "skill") {
      // The store replaces this with the expanded template.
      void runSlash(`/${entry.name}`);
    } else {
      setDraft("");
      void runSlash(`/${entry.name}`);
    }
  };

  const completeCommand = (entry: CommandEntry) => {
    setDraft(`/${entry.name} `);
    useChat.getState().requestComposerFocus();
  };

  const shownDrawer = drawerOpen
    ? {
        groups: commandGroups,
        count: commandMatches.length,
        query: slashQuery ?? "",
        activeIndex: activeCommandIndex,
      }
    : drawerCache.current;

  const commandDrawer = drawerMounted ? (
    <CommandDrawer
      groups={shownDrawer.groups}
      query={shownDrawer.query}
      activeIndex={shownDrawer.activeIndex}
      count={shownDrawer.count}
      onHover={setActiveCommand}
      onPick={pickCommand}
    />
  ) : null;

  const submit = () => {
    const text = draft.trim();
    if (!text && attachedSkills.length === 0) return;

    // The send button is another way to take the highlighted suggestion.
    if (drawerOpen && commandMatches.length > 0) {
      pickCommand(commandMatches[activeCommandIndex]);
      return;
    }

    // Slash commands run before the readiness check — /skills and /model are
    // useful precisely when the model isn't loaded.
    if (text.startsWith("/")) {
      void runSlash(text).then((handled) => {
        const state = useChat.getState();
        if (handled) {
          // A skill attaches as a chip; the slash text itself is consumed.
          if (state.draft.trim() === text) setDraft("");
          return;
        }
        if (!state.streaming && (state.status?.is_ready ?? false)) void send(text);
      });
      return;
    }

    if (streaming || !ready) return;
    void send(draft);
  };

  const placeholder = () => {
    if (!booted) return "Starting Dizel…";
    if (micState === "recording") return "Listening… click the mic to stop";
    if (micState === "transcribing") return "Transcribing…";
    if (!ready) return "Waiting for the model to load...";
    if (debug) return "Message Dizel… /skills · /help";
    return "Message Dizel… or /skills";
  };

  return (
    <div className="composer-wrap">
      {dictation && dictation.state !== "idle" && (
        <div className={dictation.state === "error" ? "dictation-bar is-error" : "dictation-bar"}>
          <Icon name="mic" size={12} />
          <span
            className="dictation-text"
            title={dictation.state === "error" ? dictation.error : undefined}
          >
            {dictation.state === "downloading" &&
              (dictation.total === 0
                ? `Preparing Whisper ${dictation.size}…`
                : `Downloading Whisper ${dictation.size} — ${dictation.percent}%`)}
            {dictation.state === "done" && `Whisper ${dictation.size} ready — click the mic`}
            {dictation.state === "error" &&
              (dictation.error
                ? `Download failed — ${dictation.error}`
                : "Dictation model download failed")}
          </span>

          {dictation.state === "downloading" && (
            <span className="dictation-track">
              <span style={{ width: `${dictation.percent}%` }} />
            </span>
          )}

          {dictation.state === "downloading" ? (
            <button className="dictation-action" onClick={() => void cancelDictation()}>
              Cancel
            </button>
          ) : dictation.state === "error" ? (
            <>
              <button className="dictation-action" onClick={() => void startDictation()}>
                Retry
              </button>
              <button className="dictation-action" onClick={dismissDictation}>
                Dismiss
              </button>
            </>
          ) : (
            <button className="dictation-action" onClick={dismissDictation}>
              Dismiss
            </button>
          )}
        </div>
      )}

      {editingId && (
        <div className="edit-bar">
          <Icon name="compose" size={12} />
          <span>Editing your message — everything after it will be replaced.</span>
          <button className="edit-bar-cancel" onClick={cancelEdit}>
            Cancel
          </button>
        </div>
      )}

      <div className="composer-stack">
        {!emptyState && drawerMounted && (
          <div
            className={`composer-drawer-host is-above${drawerClosing ? " is-closing" : ""}`}
            ref={drawerHostRef}
            aria-hidden={drawerClosing || undefined}
          >
            {commandDrawer}
          </div>
        )}

        <div className="composer" ref={composerRef}>
        {attachedSkills.length > 0 && (
          <div className="skill-chips" aria-label="Attached skills">
            {attachedSkills.map((skill, index) => (
              <div
                className="skill-chip"
                key={`${skill.name}-${index}`}
                title={`/${skill.name}${skill.args ? ` ${skill.args}` : ""} — applies when you send`}
              >
                <span className="skill-mark" aria-hidden="true">
                  ✦
                </span>
                <span className="skill-name">{skill.title}</span>
                <button
                  className="skill-x"
                  onClick={() => detachSkill(index)}
                  aria-label={`Remove ${skill.title}`}
                >
                  <Icon name="close" size={10} />
                </button>
              </div>
            ))}
          </div>
        )}
        {attachments.length > 0 && (
          <div className="chips">
            {attachments.map((file) => (
              <div
                className={openPreview?.id === file.id ? "chip open" : "chip"}
                key={file.id}
                title={`${file.path} — click to read`}
                role="button"
                tabIndex={0}
                onClick={() => setOpenPreview(file)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    setOpenPreview(file);
                  }
                }}
              >
                {file.previewUrl ? (
                  <img className="chip-thumb" src={file.previewUrl} alt="" />
                ) : (
                  <span className={`chip-badge is-${kindOf(file.name).icon}`}>
                    {extension(file.name)}
                  </span>
                )}
                <span className="chip-name">{file.name}</span>
                <span className="chip-size">{formatSize(file.size)}</span>
                {file.truncated && <span className="chip-note">trimmed</span>}
                <button
                  className="chip-x"
                  onClick={(event) => {
                    event.stopPropagation();
                    removeAttachment(file.id);
                    if (openPreview?.id === file.id) setOpenPreview(null);
                  }}
                  aria-label={`Remove ${file.name}`}
                >
                  <Icon name="close" size={10} />
                </button>
              </div>
            ))}
          </div>
        )}

        <textarea
          ref={ref}
          rows={1}
          value={draft}
          spellCheck={spellcheck}
          disabled={!booted}
          placeholder={placeholder()}
          aria-autocomplete="list"
          aria-expanded={drawerOpen}
          aria-controls={drawerOpen ? "command-drawer" : undefined}
          aria-activedescendant={
            drawerOpen && commandMatches[activeCommandIndex]
              ? `command-option-${commandMatches[activeCommandIndex].name}`
              : undefined
          }
          onChange={(event) => {
            setDraft(event.target.value);
            setDrawerDismissed(false);
          }}
          onKeyDown={(event) => {
            if (event.nativeEvent.isComposing) return;

            // Backspace on an empty draft removes the newest attached skill,
            // the same way it would delete a character.
            if (
              event.key === "Backspace" &&
              draft === "" &&
              attachedSkills.length > 0
            ) {
              event.preventDefault();
              detachSkill(attachedSkills.length - 1);
              return;
            }

            if (drawerOpen) {
              const active = commandMatches[activeCommandIndex];
              if (event.key === "ArrowDown") {
                event.preventDefault();
                setActiveCommand((activeCommandIndex + 1) % commandMatches.length);
                return;
              }
              if (event.key === "ArrowUp") {
                event.preventDefault();
                setActiveCommand(
                  (activeCommandIndex - 1 + commandMatches.length) % commandMatches.length,
                );
                return;
              }
              if (
                event.key === "PageDown" ||
                event.key === "PageUp" ||
                event.key === "Home" ||
                event.key === "End"
              ) {
                event.preventDefault();
                const last = commandMatches.length - 1;
                if (event.key === "PageDown") {
                  setActiveCommand(Math.min(last, activeCommandIndex + 6));
                } else if (event.key === "PageUp") {
                  setActiveCommand(Math.max(0, activeCommandIndex - 6));
                } else if (event.key === "Home") {
                  setActiveCommand(0);
                } else {
                  setActiveCommand(last);
                }
                return;
              }
              if (event.key === "Tab") {
                event.preventDefault();
                if (active) completeCommand(active);
                return;
              }
              if (event.key === "Escape") {
                event.preventDefault();
                setDrawerDismissed(true);
                return;
              }
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                if (active) pickCommand(active);
                return;
              }
            }

            if (event.key === "Escape" && escapeClearsDraft && draft) {
              event.preventDefault();
              setDraft("");
              return;
            }
            if (event.key === "ArrowUp" && arrowUpEditsLast && !draft.trim() && !editingId) {
              const lastUser = [...useChat.getState().messages]
                .reverse()
                .find((m) => m.role === "user");
              if (lastUser) {
                event.preventDefault();
                useChat.getState().beginEdit(lastUser.id);
              }
              return;
            }
            if (event.key !== "Enter") return;
            const modifier = event.ctrlKey || event.metaKey;
            // Enter sends by default; with the pref off, only Ctrl/Cmd+Enter does.
            const shouldSend = enterToSend ? !event.shiftKey : modifier;
            if (shouldSend) {
              event.preventDefault();
              submit();
            }
          }}
        />

        <div className="composer-bar">
          <Picker
            open={toolsOpen}
            placement={placement}
            onOpenChange={setToolsOpen}
            triggerClassName={toolsOpen ? "bar-btn is-open" : "bar-btn"}
            triggerTitle="Add to prompt"
            label={<Icon name="plus" size={18} />}
          >
            <div className="popover-label">Attach</div>
            <button
              className="popover-item"
              disabled={attachments.length >= 4}
              onClick={() => {
                setToolsOpen(false);
                void addFiles();
              }}
            >
              <span className="popover-item-main">
                <Icon name="file" size={14} /> Files…
              </span>
            </button>
            <button
              className="popover-item"
              disabled={attachments.length >= 4}
              onClick={() => {
                setToolsOpen(false);
                void addFolder();
              }}
            >
              <span className="popover-item-main">
                <Icon name="folder" size={14} /> Folder…
              </span>
            </button>
            <button
              className="popover-item"
              disabled={!visionReady || attachments.length >= 4}
              title={
                visionReady
                  ? "Attach images to the prompt"
                  : "The loaded model does not support image input"
              }
              onClick={() => {
                setToolsOpen(false);
                void addImages();
              }}
            >
              <span className="popover-item-main">
                <Icon name="image" size={14} /> Images…
              </span>
              <span className="popover-hint">
                {visionReady ? "vision" : "no vision"}
              </span>
            </button>

            {recentPaths.length > 0 && (
              <>
                <div className="popover-label">Recent</div>
                {recentPaths.slice(0, 3).map((path) => (
                  <button
                    key={path}
                    className="popover-item"
                    disabled={attachments.length >= 4}
                    title={path}
                    onClick={() => {
                      setToolsOpen(false);
                      void attachPaths([path]);
                    }}
                  >
                    <span className="popover-item-main">
                      <Icon name="file" size={14} />
                      <span className="popover-file">{fileName(path)}</span>
                    </span>
                  </button>
                ))}
              </>
            )}

            <div className="popover-sep" />
            <div className="popover-label">Reply</div>
            <VerbosityPicker open={toolsOpen} />

            <div className="popover-sep" />
            <div className="popover-label">Insert</div>
            <button
              className="popover-item"
              onClick={() => {
                setToolsOpen(false);
                void pasteText();
              }}
            >
              <span className="popover-item-main">
                <Icon name="copy" size={14} /> Paste text
              </span>
              <span className="popover-hint">clipboard</span>
            </button>
            <button
              className="popover-item"
              onClick={() => {
                setToolsOpen(false);
                insertCodeBlock();
              }}
            >
              <span className="popover-item-main">
                <Icon name="code" size={14} /> Code block
              </span>
              <span className="popover-hint">fenced</span>
            </button>
            <button
              className="popover-item"
              onClick={() => {
                setToolsOpen(false);
                setDrawerDismissed(false);
                setDraft("/");
                useChat.getState().requestComposerFocus();
              }}
            >
              <span className="popover-item-main">
                <Icon name="sparkles" size={14} /> Prompt skills
              </span>
              <span className="popover-hint">/</span>
            </button>
            <button
              className="popover-item"
              onClick={() => {
                setToolsOpen(false);
                useChat.getState().openPlugins();
              }}
            >
              <span className="popover-item-main">
                <Icon name="plug" size={14} /> Plugins
              </span>
              <span className="popover-hint">MCP</span>
            </button>

            {prompts.length > 0 && (
              <>
                <div className="popover-sep" />
                <div className="popover-label">Saved prompts</div>
                {prompts.slice(0, 3).map((prompt) => (
                  <button
                    key={prompt.id}
                    className="popover-item"
                    title={prompt.body}
                    onClick={() => {
                      setToolsOpen(false);
                      const next = draft.trim()
                        ? `${draft.replace(/\s+$/, "")}\n\n${prompt.body}`
                        : prompt.body;
                      setDraft(next);
                      window.requestAnimationFrame(() => ref.current?.focus());
                    }}
                  >
                    <span className="popover-item-main">
                      <Icon name="book" size={14} />
                      <span className="popover-file">
                        {prompt.title || "Saved prompt"}
                      </span>
                    </span>
                  </button>
                ))}
              </>
            )}

            {attachments.length > 0 && (
              <>
                <div className="popover-sep" />
                <button
                  className="popover-item"
                  onClick={() => {
                    setToolsOpen(false);
                    clearAttachments();
                  }}
                >
                  <span className="popover-item-main">
                    <Icon name="trash" size={14} /> Clear attachments
                  </span>
                  <span className="popover-hint">{attachments.length}</span>
                </button>
              </>
            )}

          </Picker>

          {!booted ? (
            <>
              {/* Before boot we don't know the model or tier yet — a skeleton
                  beats inventing "Model" and "Dizel" for a second. */}
              <Skeleton width={128} height={24} radius={999} />
              <Skeleton width={62} height={24} radius={999} />
            </>
          ) : (
            <>
          <Picker
            open={modelOpen}
            placement={placement}
            onOpenChange={setModelOpen}
            triggerClassName="bar-chip"
            triggerTitle="Model"
            triggerDataCapability="model.switch"
            label={
              <>
                {isLocal ? (
                  <ModelIcon variant="dizel" size={16} />
                ) : (
                  <ProviderAvatar
                    avatar={activeProvider?.avatar}
                    size={16}
                  />
                )}
                {modelLabel}
                <Icon name="chevronDown" size={11} />
              </>
            }
          >
            <div className="popover-label">Model</div>

            <button
              className={isLocal ? "model-row active" : "model-row"}
              onClick={() => {
                setModelOpen(false);
                if (!isLocal) void selectModel("local", "");
              }}
            >
              <span className="model-row-icon">
                <ModelIcon variant={modelFamily === "Mila" ? "mila" : "dizel"} size={22} />
              </span>
              <span className="model-row-text">
                <b>{localName}</b>
                <small>{localMeta}</small>
              </span>
              {isLocal && <Icon name="check" size={13} />}
            </button>

            {recentModels.length > 0 && (
              <>
                <div className="popover-label">Recently loaded</div>
                {recentModels.map((entry) =>
                  renderLocalEntry(entry, samePath(entry.path, loadedPath)),
                )}
              </>
            )}

            {otherModels.length > 0 && (
              <>
                <button
                  type="button"
                  className="popover-collapse"
                  aria-expanded={allModelsOpen}
                  onClick={() => setAllModelsOpen((open) => !open)}
                >
                  <span>All checkpoints</span>
                  <span className="popover-collapse-count">{otherModels.length}</span>
                  <Icon name="chevronDown" size={11} />
                </button>
                {allModelsOpen &&
                  otherModels.map((entry) =>
                    renderLocalEntry(
                      { ...entry, exists: true },
                      samePath(entry.path, loadedPath),
                    ),
                  )}
              </>
            )}

            {!isLocal && (
              <>
                <div className="popover-label-row">
                  <span className="popover-label">{activeProvider?.name ?? activeSlug}</span>
                  <button
                    className="popover-icon-btn"
                    title="Refresh models"
                    aria-label="Refresh models"
                    onClick={() => void loadProviderModels()}
                  >
                    <Icon name="refresh" size={12} />
                  </button>
                </div>

                {providerModels.map((model) => (
                  <button
                    key={model.id}
                    className={model.id === status?.api_model ? "model-row active" : "model-row"}
                    onClick={() => {
                      setModelOpen(false);
                      void selectModel(activeSlug, model.id);
                    }}
                  >
                    <span className="model-row-icon">
                      <ProviderAvatar
                        avatar={activeProvider?.avatar}
                            size={22}
                      />
                    </span>
                    <span className="model-row-text">
                      <b>{model.name}</b>
                      <small>{activeProvider?.name ?? activeSlug}</small>
                    </span>
                    {model.id === status?.api_model && <Icon name="check" size={13} />}
                  </button>
                ))}

                {providerModels.length === 0 && (
                  <div className="popover-empty">
                    {providerModelsError ?? "No models listed for this provider."}
                  </div>
                )}
              </>
            )}

            <div className="popover-sep" />
            <button
              className="popover-item"
              onClick={() => {
                setModelOpen(false);
                openSettings();
              }}
            >
              <span className="popover-item-main">Manage models…</span>
            </button>
          </Picker>

          {isLocal && (
            <ModePicker
              open={modeOpen}
              onOpenChange={setModeOpen}
              placement={placement}
              modes={modes}
              mode={mode}
              effort={thinkingEffort}
              speed={fastSpeed}
              onMode={(name) => {
                // Do not close here: ModePicker closes itself when a mode row is
                // clicked, and this same handler is reached from the effort panel,
                // where switching to Thinking must leave the panel open.
                void setMode(name);
              }}
              onEffort={(next) => {
                void setThinkingEffort(next);
              }}
              onSpeed={(next) => {
                void setFastSpeed(next);
              }}
            />
          )}


            </>
          )}

          <span className="spacer" />

          <button
            className={micState === "recording" ? "bar-btn rec" : "bar-btn"}
            onClick={() => {
              // The selected Whisper size can change at any time. Re-check it
              // before recording; a cached "ready" flag belongs to the old
              // model and would otherwise fail with a 409 mid-transcription.
              void micAction();
            }}
            disabled={micState === "transcribing"}
            title={micState === "recording" ? "Stop and transcribe" : "Dictate"}
          >
            {micState === "transcribing" ? (
              <span className="mini-spin" />
            ) : (
              <Icon name="mic" size={16} />
            )}
          </button>

          {streaming ? (
            <button className="send-circle stop" onClick={() => void stop()} title="Stop generating">
              <span className="stop-square" />
            </button>
          ) : (
            <button
              className="send-circle"
              onClick={submit}
              disabled={(!draft.trim() && attachedSkills.length === 0) || !ready}
              title="Send"
            >
              <Icon name="arrowUp" size={16} strokeWidth={2} />
            </button>
          )}
        </div>
      </div>

        {emptyState && drawerMounted && (
          <div
            className={`composer-drawer-host is-below${drawerClosing ? " is-closing" : ""}`}
            ref={drawerHostRef}
            aria-hidden={drawerClosing || undefined}
          >
            {commandDrawer}
          </div>
        )}
      </div>

      {micError ? (
        <div className="composer-hint is-error">{micError}</div>
      ) : (
        <div className={drawerMounted ? "composer-hint is-hidden" : "composer-hint"}>
          <span>{enterToSend ? "Enter to send" : "Ctrl+Enter to send"}</span>
          <i />
          <span>{enterToSend ? "Shift+Enter for new line" : "Enter for new line"}</span>
          <i />
          {draft.trimStart().startsWith("/") ? (
            <span>/skills lists prompt skills</span>
          ) : (
            <span>Dizel can make mistakes. Please verify important information.</span>
          )}
        </div>
      )}
    </div>
  );
}
