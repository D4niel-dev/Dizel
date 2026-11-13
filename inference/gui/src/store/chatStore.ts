import { create } from "zustand";

import type {
  Attachment,
  ChatStatus,
  Fork,
  Message,
  ModelEntry,
  Prefs,
  ProfileDetails,
  ProviderInfo,
  SavedPrompt,
  SessionSummary,
  SkillInfo,
  WorkspaceView,
} from "../types";
import {
  applyVariants,
  assistantTurnIndex,
  loadVariants,
  saveVariants,
} from "../lib/variants";
import type { RecentChat, DictationState, ContextUsage } from "../services/api";
import { resetSidecar } from "../services/sidecar";
import * as api from "../services/api";
import { resetApi } from "../services/api";
import { startRecording } from "../services/audio";
import { modelName } from "../services/model";
import type { Recorder } from "../services/audio";
import { CATEGORY_ORDER, DEBUG_COMMAND_NAMES } from "../services/commands";

export type ThemeName = "dark" | "light" | "dark-blue";
export type MicState = "idle" | "recording" | "transcribing";

const THEME_KEY = "dizel.theme";
const PROMPTS_KEY = "dizel.prompts";
const RECENT_ATTACHMENTS_KEY = "dizel.recent-attachments";
const THEMES: ThemeName[] = ["dark", "light", "dark-blue"];
// Local SKILL.md packs can be many KB; half of a 2k-token window is worth flagging.
const LARGE_SKILL_CHARS = 4000;

function loadRecentAttachments(): string[] {
  try {
    const raw = localStorage.getItem(RECENT_ATTACHMENTS_KEY);
    const parsed = raw ? (JSON.parse(raw) as unknown) : [];
    return Array.isArray(parsed)
      ? parsed.filter((path): path is string => typeof path === "string").slice(0, 5)
      : [];
  } catch {
    return [];
  }
}

function rememberAttachments(paths: string[]): string[] {
  const next = [...new Set([...paths, ...loadRecentAttachments()])].slice(0, 5);
  try {
    localStorage.setItem(RECENT_ATTACHMENTS_KEY, JSON.stringify(next));
  } catch {
    /* quota — recent files are a convenience */
  }
  return next;
}

function loadPrompts(): SavedPrompt[] {
  try {
    const raw = localStorage.getItem(PROMPTS_KEY);
    return raw ? (JSON.parse(raw) as SavedPrompt[]) : [];
  } catch {
    return [];
  }
}

function persistPrompts(prompts: SavedPrompt[]): void {
  try {
    localStorage.setItem(PROMPTS_KEY, JSON.stringify(prompts));
  } catch {
    /* quota — prompts are a convenience */
  }
}

const MAX_ATTACHMENTS = 4;
const MAX_ATTACHMENT_CHARS = 6000;

function initialTheme(): ThemeName {
  const stored = localStorage.getItem(THEME_KEY) as ThemeName | null;
  return stored && THEMES.includes(stored) ? stored : "dark";
}

const uid = () =>
  typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `m_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

/** Every markdown block and inline style the transcript can render. */
const DEBUG_MARKDOWN_SAMPLE = [
  "# Heading one",
  "## Heading two",
  "### Heading three",
  "#### Heading four",
  "##### Heading five",
  "###### Heading six",
  "",
  "A paragraph with **bold**, *italic*, ~~strikethrough~~, `inline code`, and a [link](https://example.com).",
  "",
  "> A blockquote with **weight**.",
  ">",
  "> A second paragraph inside the same quote.",
  "",
  "- Bullet one",
  "- Bullet two",
  "  - Nested bullet",
  "    - Deeper bullet",
  "",
  "1. First step",
  "2. Second step",
  "   1. Nested step",
  "",
  "- [x] Checked task",
  "- [ ] Unchecked task",
  "",
  "| Surface | Weight | Notes |",
  "| :--- | :---: | ---: |",
  "| Answer | Primary | Full size |",
  "| Thought | Muted | Compact |",
  "",
  "---",
  "",
  "![Provider icon](./providers/meta.png)",
  "",
  "```python",
  "def reverse(text: str) -> str:",
  '    """Return text with its characters in reverse order."""',
  "    return text[::-1]",
  "```",
].join("\n");

const DEFAULT_PREFS: Prefs = {
  fontScale: "Medium",
  showTimestamps: true,
  animations: true,
  syntaxHighlight: true,
  enterToSend: true,
  thinkingPause: "short",
  thinkingPauseCustom: 60,
  avatars: true,
  accent: "blue",
  density: "comfortable",
  clock24h: true,
  showSpeed: true,
  fontFamily: "system",
  messageWidth: "comfortable",
  separators: false,
  codeWrap: false,
  spellcheck: true,
  autoScroll: true,
  messageSpacing: "comfortable",
  showSuggestions: true,
  notifyOnFinish: true,
  lineHeight: "normal",
  underlineLinks: true,
  reduceTransparency: false,
  messageCorners: "rounded",
  showModelTag: false,
  typingAnimation: "smooth",
  autoFocusComposer: true,
  escapeClearsDraft: false,
  arrowUpEditsLast: true,
  copyLastReplyHotkey: true,
  stopGenerationHotkey: true,
};

/** Reads the appearance block into the shape the components actually consume. */
export function prefsFromConfig(appearance: Record<string, unknown> | undefined): Prefs {
  const a = appearance ?? {};
  const scale = a.font_scale as Prefs["fontScale"] | undefined;
  const accent = a.accent as Prefs["accent"] | undefined;

  return {
    fontScale: scale === "Small" || scale === "Large" ? scale : "Medium",
    showTimestamps: a.show_timestamps !== false,
    animations: a.animations !== false,
    syntaxHighlight: a.syntax_highlighting !== false,
    enterToSend: a.enter_to_send !== false,
    avatars: a.avatar_display !== "Hide avatars",
    thinkingPause:
      a.thinking_pause === "Off"
        ? "off"
        : a.thinking_pause === "Long (10s)"
          ? "long"
          : a.thinking_pause === "Custom"
            ? "custom"
            : "short",
    thinkingPauseCustom: (() => {
      const seconds = Number(a.thinking_pause_custom);
      return Number.isFinite(seconds) && seconds >= 1
        ? Math.min(86400, Math.round(seconds))
        : 60;
    })(),
    accent:
      accent === "teal" || accent === "violet" || accent === "rose" ? accent : "blue",
    density: a.density === "compact" ? "compact" : "comfortable",
    clock24h: a.clock_24h !== false,
    showSpeed: a.show_speed !== false,
    fontFamily:
      a.font_family === "serif" || a.font_family === "mono" ? a.font_family : "system",
    messageWidth:
      a.message_width === "wide" || a.message_width === "full"
        ? a.message_width
        : "comfortable",
    separators: a.separators === true,
    codeWrap: a.code_wrap === true,
    spellcheck: a.spellcheck !== false,
    autoScroll: a.auto_scroll !== false,
    messageSpacing: a.message_spacing === "compact" ? "compact" : "comfortable",
    showSuggestions: a.show_suggestions !== false,
    notifyOnFinish: a.notify_on_finish !== false,
    lineHeight: a.line_height === "relaxed" ? "relaxed" : "normal",
    underlineLinks: a.underline_links !== false,
    reduceTransparency: a.reduce_transparency === true,
    messageCorners:
      a.message_corners === "soft" || a.message_corners === "sharp"
        ? a.message_corners
        : "rounded",
    showModelTag: a.show_model_tag === true,
    typingAnimation:
      a.typing_animation === "off" ||
      a.typing_animation === "fast" ||
      a.typing_animation === "cinematic"
        ? a.typing_animation
        : "smooth",
    autoFocusComposer: a.auto_focus_composer !== false,
    escapeClearsDraft: a.escape_clears_draft === true,
    arrowUpEditsLast: a.arrow_up_edits_last !== false,
    copyLastReplyHotkey: a.copy_last_reply_hotkey !== false,
    stopGenerationHotkey: a.stop_generation_hotkey !== false,
  };
}

/** Reads the user_profile block into the shape the UI consumes. */
export function profileFromConfig(raw: Record<string, unknown> | undefined): ProfileDetails {
  const p = raw ?? {};
  const accent = p.accent;
  const style = p.greeting_style;
  return {
    bio: typeof p.bio === "string" ? p.bio : "",
    pronouns: typeof p.pronouns === "string" ? p.pronouns : "",
    accent:
      accent === "teal" || accent === "violet" || accent === "rose" || accent === "amber"
        ? accent
        : "blue",
    greetingStyle: style === "simple" || style === "time" ? style : "warm",
    createdAt: typeof p.created_at === "string" ? p.created_at : "",
  };
}

/** Keeps image payloads from ballooning the request and the session JSON. */
async function prepareImageDataUrl(source: string): Promise<string> {
  return new Promise((resolve) => {
    const image = new Image();
    image.onload = () => {
      const longest = Math.max(image.naturalWidth, image.naturalHeight);
      if (longest <= 1568) {
        resolve(source);
        return;
      }
      const scale = 1568 / longest;
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
      canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
      const context = canvas.getContext("2d");
      if (!context) {
        resolve(source);
        return;
      }
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      try {
        resolve(canvas.toDataURL("image/jpeg", 0.88));
      } catch {
        resolve(source);
      }
    };
    image.onerror = () => resolve(source);
    image.src = source;
  });
}

/** Shared by "attach files" and "attach folder". */
async function ingestPaths(
  paths: string[],
  existing: Attachment[],
  set: (partial: Partial<ChatState>) => void,
  allowImages = false,
): Promise<void> {
  const bridge = window.dizel;
  if (!bridge || paths.length === 0) return;

  const room = MAX_ATTACHMENTS - existing.length;
  if (room <= 0) {
    set({ error: `Up to ${MAX_ATTACHMENTS} attachments per message.` });
    return;
  }

  const added: Attachment[] = [];
  const failures: string[] = [];

  const IMAGE = /\.(png|jpe?g|webp|gif|bmp|svg)$/i;

  for (const path of paths.slice(0, room)) {
    const name = path.split(/[\\/]/).pop() ?? path;

    if (IMAGE.test(path)) {
      if (!allowImages) {
        failures.push(`${name} — the loaded model has no vision input`);
        continue;
      }
      const image = await bridge.readDataUrl(path);
      if (!image.ok || !image.dataUrl) {
        failures.push(`${name} — ${image.error ?? "could not be read"}`);
        continue;
      }
      added.push({
        id: uid(),
        name,
        path,
        size: image.size ?? 0,
        content: "",
        truncated: false,
        previewUrl: await prepareImageDataUrl(image.dataUrl),
      });
      continue;
    }

    const result = await bridge.readTextFile(path);

    if (!result.ok || typeof result.content !== "string") {
      failures.push(`${name} — ${result.error ?? "could not be read"}`);
      continue;
    }

    const clipped = result.content.slice(0, MAX_ATTACHMENT_CHARS);
    added.push({
      id: uid(),
      name,
      path,
      size: result.size ?? result.content.length,
      content: clipped,
      truncated: Boolean(result.truncated) || clipped.length < result.content.length,
    });
  }

  if (added.length > 0) {
    const textPaths = added.filter((item) => !item.previewUrl).map((item) => item.path);
    set({
      attachments: [...existing, ...added],
      ...(textPaths.length > 0 ? { recentPaths: rememberAttachments(textPaths) } : {}),
    });
  }
  if (failures.length > 0) {
    set({ error: failures.join(" · ") });
  }
}

/** Side effects that can't be expressed as a React prop. */
function applyPrefs(prefs: Prefs): void {
  const zoom = prefs.fontScale === "Small" ? 0.94 : prefs.fontScale === "Large" ? 1.08 : 1;
  window.dizel?.setZoom(zoom);
  document.documentElement.dataset.motion = prefs.animations ? "on" : "off";
  document.documentElement.dataset.accent = prefs.accent;
  document.documentElement.dataset.density = prefs.density;
  document.documentElement.dataset.font = prefs.fontFamily;
  document.documentElement.dataset.width = prefs.messageWidth;
  document.documentElement.dataset.separators = prefs.separators ? "on" : "off";
  document.documentElement.dataset.codewrap = prefs.codeWrap ? "on" : "off";
  document.documentElement.dataset.spacing = prefs.messageSpacing;
  document.documentElement.dataset.lineheight = prefs.lineHeight;
  document.documentElement.dataset.linkstyle = prefs.underlineLinks ? "underline" : "plain";
  document.documentElement.dataset.transparency = prefs.reduceTransparency ? "reduced" : "full";
  document.documentElement.dataset.corners = prefs.messageCorners;
}

function patchLast(messages: Message[], patch: (m: Message) => Message): Message[] {
  // The reply being streamed is the last *assistant* message, not necessarily the
  // last message. A notice arriving mid-stream (compaction) used to sit after it,
  // so every token was discarded and the bubble stayed on "…" forever.
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index].role === "assistant") {
      const next = messages.slice();
      next[index] = patch(messages[index]);
      return next;
    }
  }
  return messages;
}

/** Files are folded into the prompt as tagged blocks the model can read. */
function composePrompt(text: string, attachments: Attachment[], vision: boolean): string {
  if (attachments.length === 0) return text;
  const blocks = attachments.map((file) =>
    file.previewUrl
      ? `<file name="${file.name}" type="image">\n${
          vision
            ? "[image attached]"
            : "[image attached — this model cannot see it; ask the user to describe it if it matters]"
        }\n</file>`
      : `<file name="${file.name}">\n${file.content}${file.truncated ? "\n[truncated]" : ""}\n</file>`,
  );
  return `${blocks.join("\n\n")}\n\n${text}`;
}

interface ChatState {
  booted: boolean;
  bootError: string | null;
  status: ChatStatus | null;
  messages: Message[];
  sessionId: string | null;
  sessions: SessionSummary[];
  streaming: boolean;
  error: string | null;
  theme: ThemeName;
  sidebarCollapsed: boolean;
  showRecent: boolean;
  draft: string;
  username: string;

  models: ModelEntry[];
  profiles: Record<string, string[]>;
  variant: string;
  mode: string;
  attachments: Attachment[];
  /** Paths of the last text files attached, newest first. */
  recentPaths: string[];
  /** The file whose contents are open in the side panel, if any. */
  preview: Attachment | null;
  /** Bumped by every newChat. A counter, not a boolean, because "already empty"
   *  is a real state — pressing New Chat there must still count as a fresh start. */
  chatEpoch: number;
  /** A turn being re-asked; committed once its new reply arrives. */
  pendingFork: { prompt: string; reply: Message | null } | null;
  /** Message id a continuation should be appended to. */
  pendingContinue: string | null;
  settingsOpen: boolean;
  settingsTab: string;
  settingsError: string | null;
  micState: MicState;
  micError: string | null;
  workspaceView: WorkspaceView | null;
  prompts: SavedPrompt[];
  avatarUrl: string | null;
  /** Identity details shown in the profile modal and around the app. */
  profile: ProfileDetails;
  profileOpen: boolean;
  statusBarOpen: boolean;
  prefs: Prefs;
  setPrefs: (patch: Partial<Prefs>) => void;
  modelVersion: string;
  providers: ProviderInfo[];
  providerModels: { id: string; name: string }[];
  providerModelsFor: string;
  providerModelsError: string | null;
  loadProviderModels: () => Promise<void>;
  selectModel: (slug: string, modelId: string) => Promise<void>;

  boot: () => Promise<void>;
  setDraft: (value: string) => void;
  /** Bumped when a command places text in the composer and wants it visible. */
  composerFocus: number;
  requestComposerFocus: () => void;
  refreshStatus: () => Promise<void>;
  loadSessions: () => Promise<void>;
  loadModels: () => Promise<void>;
  openSession: (id: string) => Promise<void>;
  togglePin: (id: string) => Promise<void>;
  archiveSession: (id: string) => Promise<void>;
  recent: RecentChat[];
  loadRecent: () => Promise<void>;
  handoff: () => Promise<void>;
  editingId: string | null;
  beginEdit: (id: string) => void;
  cancelEdit: () => void;
  dictation: DictationState | null;
  dictationReady: boolean;
  startDictation: () => Promise<void>;
  cancelDictation: () => Promise<void>;
  dismissDictation: () => void;

  debug: boolean;
  debugPanel: { title: string; body: string } | null;
  toast: string | null;
  /** Prompt templates available without debug mode, invoked as /name. */
  skills: SkillInfo[];
  skillsDir: string;
  skillsProjectDir: string;
  loadSkills: () => Promise<void>;
  toggleDebug: () => void;
  notify: (message: string) => void;
  closeDebugPanel: () => void;
  runSlash: (input: string) => Promise<boolean>;
  budgetOverride: number | null;
  /** Usage as the server counts it — the only number that reflects what the model sees. */
  serverUsage: ContextUsage | null;
  /** Set when the Python sidecar exits unexpectedly — the app is unusable until it returns. */
  sidecarDown: { code: number | null; signal: string | null } | null;
  restartSidecar: () => Promise<void>;
  toolsOpen: boolean;
  openTools: () => void;
  closeTools: () => void;
  pluginsOpen: boolean;
  openPlugins: () => void;
  closePlugins: () => void;
  setBudgetOverride: (value: number | null) => void;
  renameSession: (id: string, title: string) => Promise<void>;
  removeSession: (id: string) => Promise<void>;
  send: (text: string) => Promise<void>;
  regenerate: () => Promise<void>;
  showSibling: (id: string, delta: number) => void;
  commitFork: () => void;
  commitContinue: () => void;
  continueLast: (id: string) => Promise<void>;
  showFork: (id: string, delta: number) => void;
  stop: () => Promise<void>;
  newChat: () => void;
  setTheme: (theme: ThemeName) => void;
  cycleTheme: () => void;
  toggleSidebar: () => void;
  toggleRecent: () => void;
  dismissError: () => void;

  switchModel: (path: string) => Promise<void>;
  setVariant: (variant: string) => Promise<void>;
  setMode: (mode: string) => Promise<void>;
  addFiles: () => Promise<void>;
  addFolder: () => Promise<void>;
  addImages: () => Promise<void>;
  attachPaths: (paths: string[]) => Promise<void>;
  removeAttachment: (id: string) => void;
  setPreview: (attachment: Attachment | null) => void;
  openSettings: (tab?: string) => void;
  closeSettings: () => void;
  saveSettings: (patch: Record<string, unknown>) => Promise<void>;
  toggleMic: () => Promise<void>;
  micAction: () => Promise<void>;
  openWorkspace: (view: WorkspaceView) => void;
  closeWorkspace: () => void;
  addPrompt: (title: string, body: string) => void;
  removePrompt: (id: string) => void;
  importChat: () => Promise<void>;
  openProfile: () => void;
  closeProfile: () => void;
  saveProfile: (
    username: string,
    details: ProfileDetails,
    avatar?: File,
    removeAvatar?: boolean,
  ) => Promise<void>;
  toggleStatusBar: () => void;
}

let controller: AbortController | null = null;

/** How many dropped messages we have already reported, per session. */
const trimmedNoted = new Map<string, number>();
let pollTimer: number | null = null;
let recorder: Recorder | null = null;
let dictationTimer: number | null = null;
const DEBUG_KEY = "dizel.debug";

export const useChat = create<ChatState>((set, get) => ({
  booted: false,
  bootError: null,
  status: null,
  messages: [],
  sessionId: null,
  sessions: [],
  streaming: false,
  error: null,
  theme: initialTheme(),
  sidebarCollapsed: false,
  showRecent: true,
  draft: "",
  username: "You",

  models: [],
  profiles: {},
  variant: "Dizel",
  mode: "Fast",
  attachments: [],
  recentPaths: loadRecentAttachments(),
  preview: null,
  pendingFork: null,
  pendingContinue: null,
  chatEpoch: 0,
  settingsOpen: false,
  settingsTab: "Chat",
  settingsError: null,
  micState: "idle",
  micError: null,
  workspaceView: null,
  prompts: loadPrompts(),
  avatarUrl: null,
  profile: { bio: "", pronouns: "", accent: "blue", greetingStyle: "warm", createdAt: "" },
  profileOpen: false,
  statusBarOpen: true,
  prefs: DEFAULT_PREFS,
  modelVersion: "",
  recent: [],
  editingId: null,
  dictation: null,
  dictationReady: false,
  debug: false,
  debugPanel: null,
  toast: null,
  skills: [],
  skillsDir: "",
  skillsProjectDir: "",
  budgetOverride: null,
  serverUsage: null,
  sidecarDown: null,
  toolsOpen: false,
  pluginsOpen: false,
  providers: [],
  providerModels: [],
  providerModelsFor: "",
  providerModelsError: null,

  setPrefs: (patch) => {
    const next = { ...get().prefs, ...patch };
    applyPrefs(next);
    set({ prefs: next });
  },

  setDraft: (draft) => set({ draft }),

  composerFocus: 0,
  requestComposerFocus: () =>
    set((state) => ({ composerFocus: state.composerFocus + 1 })),

  boot: async () => {
    try {
      await api.initApi();
      // Status and config are independent once the sidecar is reachable — two
      // sequential round trips here were half the pre-boot flash.
      const [status, config] = await Promise.all([api.getStatus(), api.getConfig()]);
      const profileRaw = (config.user_profile ?? {}) as Record<string, unknown>;
      const prefs = prefsFromConfig(config.appearance as Record<string, unknown> | undefined);
      applyPrefs(prefs);

      set({
        booted: true,
        status,
        bootError: null,
        username:
          typeof profileRaw.username === "string" && profileRaw.username
            ? profileRaw.username
            : "You",
        profile: profileFromConfig(profileRaw),
        prefs,
      });

      void get().loadSessions();
      void get().loadModels();
      void get().loadProviderModels();
      void get().loadRecent();
      void get().loadSkills();

      // Debug mode survives restarts; it's a client preference, not config.
      try {
        if (localStorage.getItem(DEBUG_KEY) === "1") set({ debug: true });
      } catch {
        /* storage unavailable */
      }

      // Dictation's checkpoint is ~140 MB, so we only ask whether it's there.
      void api
        .dictationStatus()
        .then((info) => set({ dictationReady: info.downloaded }))
        .catch(() => undefined);
      void api
        .listProviders()
        .then((providers) => set({ providers }))
        .catch(() => undefined);
      void api
        .getModelFacts()
        .then((facts) => {
          if (facts.version) set({ modelVersion: facts.version });
        })
        .catch(() => undefined);
      void api
        .getAvatarUrl()
        .then((url) => {
          if (url) set({ avatarUrl: url });
        })
        .catch(() => undefined);
      void api
        .listProfiles()
        .then((profiles) => set({ profiles }))
        .catch(() => undefined);

      // If a checkpoint is configured but nothing is loaded, load it — the app
      // should be usable on launch, not require a trip to Settings.
      if (!status.is_ready) {
        const checkpoint = typeof config.checkpoint === "string" ? config.checkpoint : "";
        const device = typeof config.device === "string" ? config.device : undefined;
        if (checkpoint) {
          await api.loadModel(checkpoint, device);
          set({ status: await api.getStatus() });
          startPolling(set);
        }
      }
    } catch (error) {
      set({ booted: false, bootError: (error as Error).message });
    }
  },

  refreshStatus: async () => {
    try {
      const status = await api.getStatus();
      const changedProvider = status.provider !== get().status?.provider;
      set({ status });
      // A provider switch invalidates the model list in the composer.
      if (changedProvider) void get().loadProviderModels();
    } catch {
      /* status is cosmetic; never break the UI over it */
    }
  },

  loadProviderModels: async () => {
    const slug = get().status?.provider ?? "local";
    if (slug === "local") {
      set({ providerModels: [], providerModelsFor: "local", providerModelsError: null });
      return;
    }
    try {
      const { models } = await api.listProviderModels(slug);
      set({ providerModels: models, providerModelsFor: slug, providerModelsError: null });
    } catch (error) {
      set({
        providerModels: [],
        providerModelsFor: slug,
        providerModelsError: (error as Error).message,
      });
    }
  },

  selectModel: async (slug: string, modelId: string) => {
    try {
      // An empty key keeps the stored one; only the model id changes.
      await api.setProviderKey(slug, "", modelId);
      await get().refreshStatus();
    } catch (error) {
      set({ error: (error as Error).message });
    }
  },

  loadSessions: async () => {
    try {
      set({ sessions: await api.listSessions() });
    } catch {
      /* history is optional */
    }
  },

  loadModels: async () => {
    try {
      set({ models: await api.listModels() });
    } catch {
      /* the picker falls back to the configured checkpoint */
    }
  },

  openSession: async (id: string) => {
    if (get().streaming) return;
    try {
      const session = await api.getSession(id);
      set({
        sessionId: session.id,
        error: null,
        editingId: null,
        messages: applyVariants(
          (session.messages ?? []).map((m, index) => ({
            id: `${session.id}_${index}`,
            role:
              m.role === "assistant" ? "assistant" : m.role === "system" ? "system" : "user",
            content: m.content,
            createdAt: Date.now(),
          })),
          loadVariants(session.id),
        ),
      });
    } catch (error) {
      set({ error: (error as Error).message });
    }
  },

  togglePin: async (id: string) => {
    try {
      await api.togglePinSession(id);
      await get().loadSessions();
    } catch (error) {
      set({ error: (error as Error).message });
    }
  },

  archiveSession: async (id: string) => {
    try {
      await api.archiveSession(id);
      await get().loadSessions();
    } catch (error) {
      set({ error: (error as Error).message });
    }
  },

  beginEdit: (id: string) => {
    const message = get().messages.find((item) => item.id === id);
    if (!message || message.role !== "user") return;
    set({ editingId: id, draft: message.content });
    document.querySelector<HTMLTextAreaElement>(".composer textarea")?.focus();
  },

  cancelEdit: () => set({ editingId: null, draft: "" }),

  /**
   * Fetches the Whisper checkpoint the first time dictation is used.
   *
   * The download is started automatically rather than behind a confirmation —
   * it's ~140 MB and the user already asked for dictation by pressing the mic —
   * but it always shows progress and can be cancelled.
   */
  startDictation: async () => {
    try {
      set({ dictation: await api.startDictationDownload() });
    } catch (error) {
      set({ error: (error as Error).message });
      return;
    }

    if (dictationTimer !== null) return;

    dictationTimer = window.setInterval(async () => {
      try {
        const next = await api.dictationDownload();
        set({ dictation: next, dictationReady: next.downloaded });

        if (next.state !== "downloading") {
          if (dictationTimer !== null) window.clearInterval(dictationTimer);
          dictationTimer = null;

          // "Ready" shouldn't sit there forever — it's a confirmation, not a
          // panel. Failures stay until dismissed so the message is readable.
          if (next.state === "done") {
            window.setTimeout(() => {
              if (get().dictation?.state === "done") set({ dictation: null });
            }, 4000);
          }
        }
      } catch {
        if (dictationTimer !== null) window.clearInterval(dictationTimer);
        dictationTimer = null;
      }
    }, 700);
  },

  cancelDictation: async () => {
    try {
      set({ dictation: await api.cancelDictationDownload() });
    } catch {
      /* the poller will settle it */
    }
  },

  dismissDictation: () => set({ dictation: null }),

  toggleDebug: () => {
    const next = !get().debug;
    try {
      localStorage.setItem(DEBUG_KEY, next ? "1" : "0");
    } catch {
      /* storage unavailable */
    }
    set({
      debug: next,
      debugPanel: null,
      toast: next
        ? "Debug mode on — type /help in the composer"
        : "Debug mode off",
    });
    window.setTimeout(() => {
      if (get().toast) set({ toast: null });
    }, 3600);
  },

  notify: (message) => {
    set({ toast: message });
    window.setTimeout(() => {
      if (get().toast === message) set({ toast: null });
    }, 3600);
  },

  closeDebugPanel: () => set({ debugPanel: null }),

  setBudgetOverride: (value) => set({ budgetOverride: value }),

  /**
   * Slash commands. Skills run without debug mode; the debug switch still needs
   * it. Returns false when the input isn't a command, so the composer falls
   * through to a normal send.
   *
   * Output goes to a floating panel rather than the transcript: a command card
   * in the chat would be saved to the session and sent to the model as context.
   */
  runSlash: async (input) => {
    const raw = input.trim();
    if (!raw.startsWith("/")) return false;

    const [name, ...rest] = raw.slice(1).split(/\s+/);
    const arg = rest.join(" ");
    const lower = name.toLowerCase();
    const panel = (title: string, body: string) => set({ debugPanel: { title, body } });

    // Skills and the plugin runner are always available; debug commands still
    // need debug mode.
    if (lower === "plugins") {
      get().openPlugins();
      return true;
    }

    if (lower === "skills") {
      const action = arg.trim().toLowerCase();
      if (!get().skills.length || action === "reload") await get().loadSkills();
      if (action === "reload") {
        panel("Skills", `${get().skills.length} skill(s) available.`);
        return true;
      }
      const buckets = new Map<string, SkillInfo[]>();
      for (const skill of get().skills) {
        const category = skill.category || "General";
        const bucket = buckets.get(category);
        if (bucket) bucket.push(skill);
        else buckets.set(category, [skill]);
      }
      const rank = new Map(CATEGORY_ORDER.map((name, index) => [name, index]));
      const categories = [...buckets.keys()].sort((left, right) => {
        const leftRank = rank.get(left) ?? CATEGORY_ORDER.length;
        const rightRank = rank.get(right) ?? CATEGORY_ORDER.length;
        return leftRank - rightRank || left.localeCompare(right);
      });
      if (action) {
        const needle = action.replace(/[-_]/g, " ").trim();
        const category = categories.find((name) => name.toLowerCase() === needle);
        if (!category) {
          panel(
            "Skills",
            `Unknown skill category: ${action}\nCategories: ${categories.join(", ")}`,
          );
          return true;
        }
        const entries = buckets.get(category) ?? [];
        const rows = entries.flatMap((skill) => {
          const aliases = skill.aliases.length
            ? `\n${" ".repeat(16)}aliases: ${skill.aliases.map((alias) => `/${alias}`).join(", ")}`
            : "";
          return [`/${skill.name.padEnd(14)}${skill.description}${aliases}`];
        });
        panel(
          `Skills · ${category}`,
          [
            `${entries.length} skill(s). Type /name [input] to place one in the composer.`,
            "",
            ...rows,
          ].join("\n"),
        );
        return true;
      }

      const rows = categories.map((category) => {
        const entries = buckets.get(category) ?? [];
        if (entries.length > 10) {
          return `${category.toUpperCase()} ${entries.length} · type /skills ${category.toLowerCase()}`;
        }
        const names = entries.map((skill) => `/${skill.name}`).join(", ");
        return `${category.toUpperCase()} ${entries.length} · ${names}`;
      });
      panel(
        "Skills",
        [
          `Skills · ${get().skills.length}`,
          "",
          ...(rows.length ? rows : ["No skills found."]),
          "",
          "Type / to browse and filter every command.",
          "Type /skills <category> for descriptions and aliases.",
          `Your own skills live in ${get().skillsDir || ".dizel/skills"} — ` +
            "Markdown with name and description frontmatter.",
          `Project skills: ${get().skillsProjectDir || ".agent/skills"} — ` +
            "folders with a SKILL.md.",
          "Run /skills reload after adding one.",
          ...(get().debug ? [] : ["Debug commands: enable debug, then type /help."]),
        ].join("\n"),
      );
      return true;
    }

    if (!get().skills.length) await get().loadSkills();
    const skill = get().skills.find(
      (entry) => entry.name === lower || entry.aliases.includes(lower),
    );
    // A skill may not shadow a debug command while debug is on.
    if (skill && !(get().debug && DEBUG_COMMAND_NAMES.has(lower))) {
      try {
        const expanded = await api.expandSkill(skill.name, arg);
        set({ draft: expanded.prompt });
        get().requestComposerFocus();
        if (expanded.prompt.length > LARGE_SKILL_CHARS) {
          get().notify(
            `Large skill — ~${Math.round(expanded.prompt.length / 4)} tokens. Trim it before sending.`,
          );
        }
      } catch (error) {
        panel("Skills", (error as Error).message);
      }
      return true;
    }

    if (!get().debug) return false;

    switch (lower) {
      case "help":
        panel(
          "Debug commands",
          [
            "Inspect",
            "/ctx              context breakdown from the server",
            "/model            model + provider facts",
            "/capabilities     what each surface can do, and what is missing",
            "/status           one-screen state dump",
            "/history          list the messages sent to the model",
            "/bundle           copy a diagnostic bundle to the clipboard",
            "",
            "UI & render",
            "/echo <text>      append a fake reply (render test)",
            "/markdown         append every markdown style (render test)",
            "/thought          append a Thought panel with every markdown style",
            "/filler <n>       append n words of filler (scroll test)",
            "/error <text>     raise the error banner (UI test)",
            "/simulate <state>  loading | error | streaming | off",
            "/accent <name>    blue | teal | violet | rose",
            "",
            "Session",
            "/clear            start a new chat",
            "/export           save this chat as JSON",
            "/replay           resend the last prompt",
            "/copy             copy the last reply",
            "/pin              pin or unpin this chat",
            "/archive          archive this chat",
            "/stop             stop the current generation",
            "",
            "Context & tools",
            "/budget <n>|max   force the displayed context usage",
            "/ctxfull          fill the real context window (limit test)",
            "/usage [action]   window state; fill <pct> | exhaust | reset | expire",
            "/tools [on|off]   let the model call tools (read files, grep, calc)",
            "",
            "Runtime",
            "/bench <n>        time n runs (ttft + tok/s)",
            "/reset-ui         clear local state and reload",
            "/debug            turn debug mode off",
          ].join("\n"),
        );
        return true;

      case "debug":
        get().toggleDebug();
        return true;

      case "ctx": {
        const usage = await api.contextUsage(get().sessionId);
        panel(
          "Context",
          [
            `system       ${usage.system}`,
            `conversation ${usage.conversation}`,
            `reserved     ${usage.reserved} (${usage.task || "no reply yet"})`,
            `used         ${usage.used} / ${usage.total}  (${usage.percent}%)`,
            `counting     ${usage.exact ? "exact — model tokenizer" : "estimated — chars/4"}`,
          ].join("\n"),
        );
        return true;
      }

      case "model": {
        const info = get().status?.model_info ?? {};
        panel(
          "Model",
          [
            `name       ${modelName(get().status)}`,
            `provider   ${get().status?.provider ?? "—"}`,
            `params     ${info.params ?? "—"}`,
            `context    ${info.ctx_len ?? "—"}`,
            `device     ${info.device ? String(info.device).toUpperCase() : "—"}`,
            `tier/mode  ${get().variant} · ${get().mode}`,
            `session    ${get().sessionId ?? "unsaved"}`,
            `messages   ${get().messages.length}`,
          ].join("\n"),
        );
        return true;
      }

      case "clear":
        get().newChat();
        panel("Cleared", "Started a new chat.");
        return true;

      case "export": {
        const id = get().sessionId;
        if (!id) {
          panel("Export", "Nothing to export — this chat has no session yet.");
          return true;
        }
        const data = await api.getSession(id);
        const safe = data.title.replace(/[^\w\-. ]+/g, "").trim().slice(0, 48) || "chat";
        const saved = await window.dizel?.saveJson(
          `${safe}.json`,
          JSON.stringify(data, null, 2),
        );
        panel("Export", saved ? `Saved to ${saved}` : "Cancelled.");
        return true;
      }

      case "echo":
        set((state) => ({
          messages: [
            ...state.messages,
            {
              id: uid(),
              role: "assistant" as const,
              content: arg || "Echo with no argument.",
              createdAt: Date.now(),
            },
          ],
        }));
        panel("Echo", "Appended a fake assistant message (not saved).");
        return true;

      case "markdown":
      case "md":
        set((state) => ({
          messages: [
            ...state.messages,
            {
              id: uid(),
              role: "assistant" as const,
              content: DEBUG_MARKDOWN_SAMPLE,
              createdAt: Date.now(),
              durationMs: 4200,
            },
          ],
        }));
        panel(
          "Markdown",
          "Appended a fake assistant reply with every markdown style (not saved).",
        );
        return true;

      case "thought":
      case "think":
        set((state) => ({
          messages: [
            ...state.messages,
            {
              id: uid(),
              role: "assistant" as const,
              content:
                "A thought fixture is above. Open **Thought for 12s** to inspect every markdown style inside the Thought UI.",
              thinking: DEBUG_MARKDOWN_SAMPLE,
              thoughtMs: 12400,
              createdAt: Date.now(),
              durationMs: 2400,
            },
          ],
        }));
        panel(
          "Thought",
          "Appended a fake reply with a full-markdown Thought panel (not saved).",
        );
        return true;

      case "ctxfull": {
        // Fills the conversation with roughly a full context window so the *real*
        // limit is hit on the next send — /budget max only fakes the meter.
        const total = get().status?.model_info?.ctx_len ?? 2048;
        const chars = Math.min(240_000, Math.round(total * 4.4));
        const body = "The quick brown fox jumps over the lazy dog. "
          .repeat(Math.ceil(chars / 45) + 1)
          .slice(0, chars);

        set((state) => ({
          messages: [
            ...state.messages,
            {
              id: uid(),
              role: "assistant" as const,
              content: `**Context filler — about ${total} tokens**\n\n${body}`,
              createdAt: Date.now(),
            },
          ],
        }));
        panel(
          "Context",
          `Added roughly ${total} tokens of filler (${chars} chars).\n` +
            "Send any message to hit the real limit. /budget max only fakes the meter.",
        );
        return true;
      }

      case "filler": {
        const words = Math.min(4000, Math.max(10, Number(arg) || 400));
        const body = Array.from({ length: words }, (_, i) => `word${i + 1}`).join(" ");
        set((state) => ({
          messages: [
            ...state.messages,
            {
              id: uid(),
              role: "assistant" as const,
              content: `**${words} words of filler**\n\n${body}`,
              createdAt: Date.now(),
            },
          ],
        }));
        panel("Filler", `Appended ${words} words (not saved).`);
        return true;
      }

      case "error":
        set({ error: arg || "Test error raised by /error" });
        panel("Error", "Raised the error banner.");
        return true;

      case "replay": {
        const lastUser = [...get().messages].reverse().find((m) => m.role === "user");
        if (!lastUser) {
          panel("Replay", "No user message to replay.");
          return true;
        }
        panel("Replay", `Re-sending: ${lastUser.content.slice(0, 80)}`);
        set({ debugPanel: null });
        void get().send(lastUser.content);
        return true;
      }

      case "simulate": {
        const mode = arg.trim().toLowerCase();
        const status = get().status;
        if (mode === "loading") {
          set({
            status: {
              ...(status as ChatStatus),
              is_ready: false,
              load: {
                status: "loading",
                message: "Simulated load — /simulate off to clear",
                error: "",
                checkpoint: "simulated.gguf",
                is_ready: false,
                display_name: "Simulated Model",
              },
            },
          });
          panel("Simulate", "Forced the loading state. Run /simulate off to restore.");
        } else if (mode === "error") {
          set({ error: "Simulated failure — /simulate off to clear" });
          panel("Simulate", "Raised the error banner.");
        } else if (mode === "streaming") {
          set({ streaming: true });
          panel("Simulate", "Forced streaming=true. Send anything to clear it.");
        } else if (mode === "off") {
          const fresh = await api.getStatus();
          set({ status: fresh, error: null, streaming: false });
          panel("Simulate", "Restored the real status.");
        } else {
          panel("Simulate", "Usage: /simulate loading | error | streaming | off");
        }
        return true;
      }

      case "capabilities": {
        try {
          const data = await api.capabilities();
          panel("Capabilities", `${data.table ?? ""}\n\n${data.report}`);
          return true;
        } catch (error) {
          panel("Capabilities", (error as Error).message);
          return true;
        }
      }

      case "tools": {
        const action = arg.trim().toLowerCase();
        try {
          if (!action || action === "list") {
            // The sheet is the discoverable surface; the command just opens it.
            get().openTools();
            return true;
          }
          const enabled = action === "on" || action === "enable" || action === "true";
          const state = await api.setTools(enabled);
          panel(
            "Tools",
            enabled
              ? `Tool calls enabled — ${state.available.length} available.\n` +
                "The model can now ask to read files, search, or calculate."
              : "Tool calls disabled. Prompts are back to their normal size.",
          );
          return true;
        } catch (error) {
          panel("Tools", (error as Error).message);
          return true;
        }
      }

      case "usage": {
        // Debug surface for the 12-hour window: fill it, exhaust it, expire it.
        const [action, raw] = arg.trim().toLowerCase().split(/\s+/);
        try {
          if (!action) {
            const window_ = await api.usageWindow();
            panel(
              "Usage window",
              [
                `used        ${window_.used.toLocaleString()} / ${window_.capacity.toLocaleString()} (${window_.percent}%)`,
                `resets in   ${window_.resets_in}`,
                `exhausted   ${window_.exhausted ? "yes — sending is blocked" : "no"}`,
                `compactions ${window_.compactions_used} used, ${window_.compactions_remaining} left`,
                "",
                "usage fill <pct>   raise usage to trigger compaction",
                "usage exhaust      block sending",
                "usage reset        back to zero",
                "usage expire       pretend the window elapsed",
              ].join("\n"),
            );
            return true;
          }
          if (action === "fill") {
            const pct = Number(raw);
            const next = await api.usageDebug("fill", Number.isFinite(pct) ? pct : 90);
            panel(
              "Usage window",
              `Usage forced to ${next.percent}%.\n` +
                "Send a message — it should compact and drop a notice in the chat.",
            );
            return true;
          }
          if (action === "exhaust" || action === "reset" || action === "expire") {
            const next = await api.usageDebug(action);
            panel(
              "Usage window",
              `\`${action}\` applied — ${next.used.toLocaleString()} / ${next.capacity.toLocaleString()} ` +
                `(${next.percent}%), resets in ${next.resets_in}.\n` +
                (action === "exhaust"
                  ? "Send a message — it should refuse with the spent message."
                  : action === "expire"
                    ? "Send a message — usage should have reset to zero first."
                    : "Back to a clean window."),
            );
            return true;
          }
          panel("Usage window", `Unknown action \`${action}\`.`);
          return true;
        } catch (error) {
          panel("Usage window", (error as Error).message);
          return true;
        }
      }

      case "budget": {
        const total = get().status?.model_info?.ctx_len ?? 2048;
        if (!arg.trim() || arg.trim() === "off") {
          set({ budgetOverride: null });
          panel("Budget", "Visual override cleared — showing real usage.");
          return true;
        }
        if (arg.trim() === "max") {
          set({ budgetOverride: total });
          panel(
            "Budget",
            `Visual usage forced to ${total} / ${total} (100%).\nThe amber warning should now show. /budget off to clear.`,
          );
          return true;
        }
        const value = Number(arg);
        if (!Number.isFinite(value) || value < 0) {
          panel("Budget", "Usage: /budget <tokens> | max | off");
          return true;
        }
        set({ budgetOverride: Math.round(value) });
        panel(
          "Budget",
          `Visual usage forced to ${Math.round(value)} / ${total} (${Math.min(
            100,
            Math.round((value / total) * 100),
          )}%).\nThis is display only — the model still gets its real budget. /budget off to clear.`,
        );
        return true;
      }

      case "copy": {
        const lastReply = [...get().messages].reverse().find((m) => m.role === "assistant");
        if (!lastReply) {
          panel("Copy", "No reply to copy yet.");
          return true;
        }
        try {
          await navigator.clipboard.writeText(lastReply.content);
          panel("Copy", `Copied ${lastReply.content.length} characters.`);
        } catch {
          panel("Copy", "Clipboard unavailable.");
        }
        return true;
      }

      case "pin": {
        const id = get().sessionId;
        if (!id) {
          panel("Pin", "No session open.");
          return true;
        }
        await get().togglePin(id);
        const now = get().sessions.find((x) => x.id === id)?.pinned;
        panel("Pin", now ? "Pinned this chat." : "Unpinned this chat.");
        return true;
      }

      case "archive": {
        const id = get().sessionId;
        if (!id) {
          panel("Archive", "No session open.");
          return true;
        }
        await get().archiveSession(id);
        panel("Archive", "Archived. Find it under Workspaces → Archived.");
        return true;
      }

      case "stop":
        if (!get().streaming) {
          panel("Stop", "Nothing is generating.");
          return true;
        }
        await get().stop();
        panel("Stop", "Stopped generation.");
        return true;

      case "accent": {
        const value = arg.trim().toLowerCase();
        const allowed = ["blue", "teal", "violet", "rose"];
        if (!allowed.includes(value)) {
          panel("Accent", `Usage: /accent ${allowed.join(" | ")}`);
          return true;
        }
        get().setPrefs({ accent: value as Prefs["accent"] });
        panel("Accent", `Accent set to ${value}.`);
        return true;
      }

      case "status": {
        const st = get();
        const info = st.status?.model_info ?? {};
        panel(
          "Status",
          [
            `ready        ${st.status?.is_ready ?? false}`,
            `streaming    ${st.streaming}`,
            `provider     ${st.status?.provider ?? "—"}`,
            `model        ${modelName(st.status)}`,
            `ctx          ${info.ctx_len ?? "—"}${info.ctx_assumed ? " (assumed)" : ""}`,
            `session      ${st.sessionId ?? "unsaved"}`,
            `messages     ${st.messages.length}`,
            `tier / mode  ${st.variant} / ${st.mode}`,
            `debug        ${st.debug}`,
            `budget ovr   ${st.budgetOverride ?? "off"}`,
            `dictation    ${st.dictationReady ? "ready" : "not downloaded"}`,
            `attachments  ${st.attachments.length}`,
          ].join("\n"),
        );
        return true;
      }

      case "history": {
        const rows = get().messages.map(
          (m, i) => `${String(i).padStart(2, "0")} ${m.role.padEnd(9)} ${m.content.length
            .toString()
            .padStart(5)} chars  ${m.content.slice(0, 40).replace(/\n/g, " ")}`,
        );
        panel(
          "Messages sent to the model",
          rows.length
            ? [`${rows.length} message(s), oldest first:`, "", ...rows].join("\n")
            : "No messages yet.",
        );
        return true;
      }

      case "reset-ui":
        try {
          localStorage.clear();
        } catch {
          /* storage unavailable */
        }
        panel("Reset UI", "Cleared local state. Reloading…");
        window.setTimeout(() => window.location.reload(), 600);
        return true;

      case "bench": {
        const runs = Math.min(10, Math.max(1, Number(arg) || 3));
        const prompt = "Explain what a token is in one sentence.";
        const lines: string[] = [];
        for (let i = 1; i <= runs; i += 1) {
          const started = performance.now();
          let first = 0;
          let text = "";
          const controller = new AbortController();
          try {
            await api.streamChat(
              prompt,
              null,
              (event) => {
                if (event.type === "token") {
                  if (!first) first = performance.now();
                  text += event.text;
                }
              },
              controller.signal,
            );
            const total = performance.now() - started;
            const tokens = Math.round(text.length / 4);
            lines.push(
              `run ${i}: ttft ${first ? Math.round(first - started) : "—"}ms · ` +
                `${(total / 1000).toFixed(1)}s · ~${Math.round(tokens / (total / 1000))} tok/s`,
            );
          } catch (error) {
            lines.push(`run ${i}: failed — ${(error as Error).message}`);
          }
        }
        panel(`Bench (${runs} run${runs === 1 ? "" : "s"})`, lines.join("\n"));
        return true;
      }

      case "bundle": {
        const info = get().status?.model_info ?? {};
        const usage = await api.contextUsage(get().sessionId);
        const bundle = {
          generated: new Date().toISOString(),
          app: { version: window.dizel ? "desktop" : "web" },
          model: {
            name: modelName(get().status),
            params: info.params,
            ctx: info.ctx_len,
            device: info.device,
          },
          tier: get().variant,
          mode: get().mode,
          provider: get().status?.provider,
          session: get().sessionId,
          messages: get().messages.length,
          context: usage,
          prefs: get().prefs,
        };
        try {
          await navigator.clipboard.writeText(JSON.stringify(bundle, null, 2));
          panel("Bundle", "Copied a diagnostic bundle to the clipboard.");
        } catch {
          panel("Bundle", "Clipboard unavailable.");
        }
        return true;
      }

      default:
        panel("Unknown command", `No such command: /${name}\nType /help for the list.`);
        return true;
    }
  },

  loadRecent: async () => {
    try {
      set({ recent: await api.recentSummaries(3) });
    } catch {
      /* recall is a nicety; never break the welcome screen over it */
    }
  },

  loadSkills: async () => {
    try {
      const { skills, user_dir, project_dir } = await api.listSkills();
      set({ skills, skillsDir: user_dir, skillsProjectDir: project_dir });
    } catch {
      /* the sidecar may still be starting; a direct /skill call retries */
    }
  },

  /**
   * Carries the current conversation into a fresh chat.
   *
   * The model's window is 2048 tokens, so a long thread eventually starts
   * dropping its own history. Handing off keeps a summary of what came before
   * instead of letting the beginning vanish.
   */
  handoff: async () => {
    const { messages, streaming } = get();
    if (streaming || messages.length === 0) return;

    try {
      const { summary } = await api.summarizeMessages(
        messages.map((message) => ({ role: message.role, content: message.content })),
      );
      const note = summary || "Earlier conversation context was carried over.";

      const { session_id } = await api.createSession(
        [{ role: "system", content: `Carried over from an earlier chat. ${note}` }],
        "Continued chat",
      );

      await get().loadSessions();
      await get().loadRecent();
      await get().openSession(session_id);
    } catch (error) {
      set({ error: (error as Error).message });
    }
  },

  renameSession: async (id: string, title: string) => {
    const clean = title.trim().slice(0, 80);
    if (!clean) return;
    try {
      await api.renameSession(id, clean);
      await get().loadSessions();
    } catch (error) {
      set({ error: (error as Error).message });
    }
  },

  removeSession: async (id: string) => {
    try {
      await api.deleteSession(id);
      // Clear the open chat first so it can't keep streaming into a deleted file.
      if (get().sessionId === id) get().newChat();
      await get().loadSessions();
    } catch (error) {
      set({ error: (error as Error).message });
    }
  },

  send: async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || get().streaming) return;

    // Edit-and-resend: rewind the saved session so the replaced message and
    // everything after it are gone before the new text goes out.
    const editingId = get().editingId;
    if (editingId) {
      const index = get().messages.findIndex((message) => message.id === editingId);
      const sessionId = get().sessionId;
      if (index >= 0 && sessionId) {
        try {
          await api.truncateSession(sessionId, index);
        } catch (error) {
          set({ error: (error as Error).message, editingId: null });
          return;
        }
        // Capture the pair before the truncation removes it — this is the whole
        // point of forking: the old answer survives the edit.
        const oldUser = get().messages[index];
        const oldReply = get().messages[index + 1] ?? null;
        set({
          pendingFork: {
            prompt: oldUser?.content ?? "",
            reply: oldReply && oldReply.role === "assistant" ? oldReply : null,
          },
        });
        set((state) => ({ messages: state.messages.slice(0, index), editingId: null }));
      } else {
        set({ editingId: null });
      }
    }

    const attachments = get().attachments;
    const visionReady = Boolean(get().status?.model_info?.vision);
    const payload = composePrompt(trimmed, attachments, visionReady);
    const imagePayload = visionReady
      ? attachments
          .map((file) => file.previewUrl)
          .filter((url): url is string => Boolean(url))
      : [];

    // A longer question earns a longer "thinking" beat, but the floor never
    // delays a one-line answer by more than a second.
    const pauseMode = get().prefs.thinkingPause;
    const customHoldMs = Math.min(
      86_400_000,
      Math.max(1000, Math.round(get().prefs.thinkingPauseCustom * 1000)),
    );
    const holdMs =
      pauseMode === "off"
        ? 0
        : pauseMode === "custom"
          ? customHoldMs
          : pauseMode === "long"
            ? 10000
            : Math.min(4000, Math.max(800, Math.round(trimmed.length * 12)));

    const userMessage: Message = {
      id: uid(),
      role: "user",
      content: trimmed,
      createdAt: Date.now(),
      // Kept for display only — composePrompt already folded the text into the
      // payload the model receives.
      attachments: attachments.length ? attachments.map((file) => ({ ...file })) : undefined,
    };
    const assistantMessage: Message = {
      id: uid(),
      role: "assistant",
      content: "",
      createdAt: Date.now(),
      streaming: true,
      holdMs,
    };

    set((state) => ({
      messages: [...state.messages, userMessage, assistantMessage],
      streaming: true,
      error: null,
      draft: "",
      attachments: [],
      preview: null,
    }));

    controller = new AbortController();
    const startedAt = Date.now();

    // Thinking duration is measured from the first reasoning delta to the first
    // answer token, where the chip's "Thought for…" label comes from.
    let thoughtStartedAt: number | null = null;
    let thoughtMs: number | undefined;
    const reasoningElapsed = () => {
      if (thoughtMs !== undefined) return thoughtMs;
      if (thoughtStartedAt === null) return undefined;
      thoughtMs = Date.now() - thoughtStartedAt;
      return thoughtMs;
    };

    try {
      await api.streamChat(
        payload,
        get().sessionId,
        (event) => {
          switch (event.type) {
            case "start":
              set({ sessionId: event.session_id });
              break;
            case "thinking":
              // Same target as "token" — patchLast finds the assistant message
              // being streamed, so reasoning and answer land on one bubble.
              if (thoughtStartedAt === null) thoughtStartedAt = Date.now();
              set((state) => ({
                messages: patchLast(state.messages, (m) => ({
                  ...m,
                  thinking: (m.thinking ?? "") + event.text,
                  thoughtStartedAt: m.thoughtStartedAt ?? thoughtStartedAt ?? undefined,
                })),
              }));
              break;
            case "token":
              set((state) => ({
                messages: patchLast(state.messages, (m) => ({
                  ...m,
                  content: m.content + event.text,
                  thoughtMs: reasoningElapsed() ?? m.thoughtMs,
                })),
              }));
              break;
            case "done":
              set((state) => ({
                messages: patchLast(state.messages, (m) => ({
                  ...m,
                  content: event.text || m.content,
                  streaming: false,
                  durationMs: Date.now() - startedAt,
                  thoughtMs: reasoningElapsed() ?? m.thoughtMs,
                })),
                sessionId: event.session_id ?? state.sessionId,
              }));
              // A long answer often lands while you are in another app. The
              // TUI rings a bell; a window can only flash its taskbar button.
              if (document.hidden && get().prefs.notifyOnFinish) {
                window.dizel?.attention();
              }
              break;
            case "notice":
              // Server-side housekeeping the user should see, e.g. compaction.
              set((state) => ({
                messages: [
                  ...state.messages,
                  {
                    id: uid(),
                    role: "system" as const,
                    content: event.text,
                    createdAt: Date.now(),
                  },
                ],
              }));
              break;
            case "error":
              set((state) => ({
                error: event.message,
                messages: patchLast(state.messages, (m) => ({
                  ...m,
                  streaming: false,
                  failed: true,
                  durationMs: Date.now() - startedAt,
                  thoughtMs: reasoningElapsed() ?? m.thoughtMs,
                  content: m.content || event.message,
                })),
              }));
              break;
          }
        },
        controller.signal,
        Boolean(get().pendingContinue),
        imagePayload,
      );
    } catch (error) {
      const aborted = (error as Error).name === "AbortError";
      set((state) => ({
        error: aborted ? null : (error as Error).message,
        messages: patchLast(state.messages, (m) =>
          m.streaming
            ? {
                ...m,
                streaming: false,
                failed: !aborted,
                durationMs: Date.now() - startedAt,
                thoughtMs: reasoningElapsed() ?? m.thoughtMs,
                content: m.content || (aborted ? "" : (error as Error).message),
              }
            : m,
        ),
      }));
    } finally {
      controller = null;
      set({ streaming: false });
      get().commitFork();
      get().commitContinue();
      void get().refreshStatus();

      // The server truncates to fit rather than erroring, so a reply can arrive
      // having silently ignored earlier turns. Ask what it dropped and say so —
      // only when the count grows, so a long chat does not stack notices.
      void (async () => {
        try {
          const sessionId = get().sessionId;
          const usage = await api.contextUsage(sessionId);
          set({ serverUsage: usage });
          const dropped = usage.trimmed ?? 0;
          const noted = trimmedNoted.get(sessionId ?? "");
          if (dropped > 0 && dropped !== noted) {
            trimmedNoted.set(sessionId ?? "", dropped);
            set((state) => ({
              messages: [
                ...state.messages,
                {
                  id: uid(),
                  role: "system" as const,
                  content:
                    `Context limit reached — the model did not see the ` +
                    `${dropped} oldest message${dropped === 1 ? "" : "s"} of this chat. ` +
                    "Start a new chat to give it the full window again.",
                  createdAt: Date.now(),
                },
              ],
            }));
          }
        } catch {
          /* usage is a nicety; never fail a reply over it */
        }
      })();
      void get().loadSessions();
    }
  },

  regenerate: async () => {
    const { messages, streaming } = get();
    if (streaming) return;

    // Drop trailing assistant turns, then re-send the last user text.
    let end = messages.length;
    while (end > 0 && messages[end - 1].role === "assistant") end -= 1;
    if (end === 0 || messages[end - 1].role !== "user") return;

    const prompt = messages[end - 1].content;
    const replaced = messages.slice(end);
    const previous = replaced[0]?.siblings ?? (replaced[0] ? [replaced[0]] : []);

    set({ messages: messages.slice(0, end - 1) });
    await get().send(prompt);

    // Keep the old reply instead of destroying it: the new message becomes a
    // sibling of the one it replaced, and the bubble gains a ‹ › switcher.
    if (previous.length === 0) return;
    set((state) => {
      const next = state.messages.slice();
      const last = next[next.length - 1];
      if (!last || last.role !== "assistant") return {};
      const clean = previous.map(({ siblings: _s, siblingIndex: _i, ...rest }) => rest);
      next[next.length - 1] = {
        ...last,
        siblings: [...clean, last],
        siblingIndex: clean.length,
      };
      return { messages: next };
    });

    const sessionId = get().sessionId;
    const after = get().messages;
    const lastId = after[after.length - 1]?.id;
    const turn = lastId ? assistantTurnIndex(after, lastId) : -1;
    if (sessionId && turn >= 0) {
      const map = loadVariants(sessionId);
      map[String(turn)] = after[after.length - 1].siblings ?? [];
      saveVariants(sessionId, map);
    }
  },

  continueLast: async (id: string) => {
    if (get().streaming) return;
    set({ pendingContinue: id });
    await get().send(
      "Continue exactly where you stopped. Do not repeat any text you already wrote.",
    );
  },

  commitContinue: () => {
    const targetId = get().pendingContinue;
    if (!targetId) return;
    set({ pendingContinue: null });

    set((state) => {
      const next = state.messages.slice();
      const last = next[next.length - 1];
      const synthetic = next[next.length - 2];
      if (!last || last.role !== "assistant" || last.failed) return {};
      if (!synthetic || synthetic.role !== "user") return {};

      const targetIndex = next.findIndex((message) => message.id === targetId);
      if (targetIndex < 0) return {};

      // Join the two halves into one reply and drop the synthetic turn, so the
      // transcript reads as a single answer rather than two.
      const joined = `${next[targetIndex].content.trimEnd()}\n\n${last.content.trim()}`;
      next[targetIndex] = { ...next[targetIndex], content: joined };
      return { messages: next.slice(0, -2) };
    });
  },

  commitFork: () => {
    const pending = get().pendingFork;
    if (!pending) return;
    set({ pendingFork: null });

    set((state) => {
      const next = state.messages.slice();
      const last = next[next.length - 1];
      const userIndex = next.length - 2;
      const user = next[userIndex];
      if (!last || last.role !== "assistant" || !user || user.role !== "user") return {};

      const base = user.forks ?? [];
      // First edit: the old pair is not in the list yet, so seed it.
      const forks: Fork[] =
        base.length > 0
          ? [...base, { prompt: user.content, reply: last }]
          : [...base, { prompt: pending.prompt, reply: pending.reply }, { prompt: user.content, reply: last }];

      next[userIndex] = { ...user, forks, forkIndex: forks.length - 1 };
      return { messages: next };
    });
  },

  showFork: (id: string, delta: number) => {
    set((state) => {
      const index = state.messages.findIndex((message) => message.id === id);
      const user = state.messages[index];
      if (index < 0 || !user?.forks?.length) return {};

      const count = user.forks.length;
      const k = (((user.forkIndex ?? 0) + delta) % count + count) % count;
      const next = state.messages.slice();
      next[index] = { ...user, content: user.forks[k].prompt, forkIndex: k };

      const reply = user.forks[k].reply;
      if (reply && next[index + 1]?.role === "assistant") {
        next[index + 1] = { ...reply, id: next[index + 1].id };
      }
      return { messages: next };
    });
  },

  showSibling: (id: string, delta: number) => {
    set((state) => ({
      messages: state.messages.map((message) => {
        if (message.id !== id || !message.siblings?.length) return message;
        const count = message.siblings.length;
        const index = (((message.siblingIndex ?? 0) + delta) % count + count) % count;
        return { ...message, siblingIndex: index };
      }),
    }));
  },

  stop: async () => {
    controller?.abort();
    try {
      await api.stopGeneration();
    } catch {
      /* the abort already stopped the client side */
    }
  },

  openTools: () => set({ toolsOpen: true }),
  closeTools: () => set({ toolsOpen: false }),
  openPlugins: () => set({ pluginsOpen: true }),
  closePlugins: () => set({ pluginsOpen: false }),

  restartSidecar: async () => {
    const bridge = window.dizel;
    if (!bridge?.restartSidecar) return;
    set({ error: null });
    const result = await bridge.restartSidecar();
    if (!result.ok) {
      set({ error: result.error ?? "Could not restart the model process." });
      return;
    }
    // The port and token are new, so both caches have to go.
    resetApi();
    resetSidecar();
    set({ sidecarDown: null, booted: false });
    await get().boot();
  },

  newChat: () => {
    controller?.abort();
    set({
      chatEpoch: get().chatEpoch + 1,
      messages: [],
      sessionId: null,
      error: null,
      streaming: false,
      attachments: [],
      editingId: null,
      serverUsage: null,
      // New chat belongs to the chat surface, not whichever workspace panel
      // happened to be open when it was pressed.
      workspaceView: null,
    });
  },

  setTheme: (theme) => {
    localStorage.setItem(THEME_KEY, theme);
    document.documentElement.dataset.theme = theme;
    set({ theme });
  },

  cycleTheme: () => {
    const index = THEMES.indexOf(get().theme);
    get().setTheme(THEMES[(index + 1) % THEMES.length]);
  },

  toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),

  toggleRecent: () => set((state) => ({ showRecent: !state.showRecent })),

  dismissError: () => set({ error: null, micError: null }),

  switchModel: async (path: string) => {
    try {
      set({ error: null });
      await api.loadModel(path);
      await api.saveConfig({ checkpoint: path });
      set({ status: await api.getStatus() });
      startPolling(set);
    } catch (error) {
      set({ error: (error as Error).message });
    }
  },

  setVariant: async (variant: string) => {
    set({ variant });
    try {
      await api.applyProfile(variant, get().mode);
    } catch (error) {
      set({ error: (error as Error).message });
    }
  },

  setMode: async (mode: string) => {
    set({ mode });
    try {
      await api.applyProfile(get().variant, mode);
    } catch (error) {
      set({ error: (error as Error).message });
    }
  },

  addFiles: async () => {
    const bridge = window.dizel;
    if (!bridge) return;

    let paths: string[] = [];
    try {
      paths = await bridge.pickFiles("attachment");
    } catch {
      return;
    }

    await ingestPaths(paths, get().attachments, set);
  },

  addFolder: async () => {
    const bridge = window.dizel;
    if (!bridge) return;

    let paths: string[] = [];
    try {
      paths = await bridge.pickFolder();
    } catch {
      return;
    }

    if (paths.length === 0) {
      set({ error: "No readable text files in that folder." });
      return;
    }

    await ingestPaths(paths, get().attachments, set);
  },

  addImages: async () => {
    if (!get().status?.model_info?.vision) {
      set({
        error: "The loaded model does not support image input.",
      });
      return;
    }

    const bridge = window.dizel;
    if (!bridge) return;

    let paths: string[] = [];
    try {
      paths = await bridge.pickFiles("image");
    } catch {
      return;
    }

    await ingestPaths(paths, get().attachments, set, true);
  },

  attachPaths: async (paths) => {
    await ingestPaths(
      paths,
      get().attachments,
      set,
      Boolean(get().status?.model_info?.vision),
    );
  },

  setPreview: (attachment) => set({ preview: attachment }),

  removeAttachment: (id) =>
    set((state) => ({ attachments: state.attachments.filter((a) => a.id !== id) })),

  openSettings: (tab) =>
    set({ settingsOpen: true, settingsTab: tab || "Chat", settingsError: null }),

  closeSettings: () => set({ settingsOpen: false }),

  saveSettings: async (patch) => {
    const before = get().status;
    const previousPath = before?.load?.checkpoint ?? "";
    const previousDevice = before?.model_info?.device ?? "";
    const nextPath = typeof patch.checkpoint === "string" ? patch.checkpoint : previousPath;
    const nextDevice = typeof patch.device === "string" ? patch.device : previousDevice;

    set({ settingsError: null });

    try {
      await api.saveConfig(patch);
      await api.reloadSettings();
      set({ status: await api.getStatus(), settingsOpen: false, error: null });

      const theme = (patch.appearance as { theme?: string } | undefined)?.theme;
      if (theme && (THEMES as string[]).includes(theme)) {
        get().setTheme(theme as ThemeName);
      }

      // Apply display preferences immediately rather than waiting for a restart.
      const appearance = patch.appearance as Record<string, unknown> | undefined;
      if (appearance) {
        const nextPrefs = prefsFromConfig(appearance);
        applyPrefs(nextPrefs);
        set({ prefs: nextPrefs });
      }

      // Saving a different checkpoint (or device) has to actually load it —
      // the dialog tells the user it does.
      if (nextPath && (nextPath !== previousPath || nextDevice !== previousDevice)) {
        await api.loadModel(nextPath, nextDevice || undefined);
        set({ status: await api.getStatus() });
        startPolling(set);
      }
    } catch (error) {
      // Keep the dialog open — a failed save behind a closing modal is invisible.
      set({ settingsError: (error as Error).message });
    }
  },

  micAction: async () => {
    if (get().micState !== "idle") {
      await get().toggleMic();
      return;
    }

    try {
      const status = await api.dictationStatus();
      set({ dictationReady: status.downloaded });

      if (!status.available) {
        set({
          micError:
            status.reason || "Whisper is not installed on this machine.",
        });
        return;
      }

      if (!status.downloaded) {
        set({ micError: null });
        await get().startDictation();
        return;
      }

      await get().toggleMic();
    } catch (error) {
      set({
        micError: `Could not check dictation status: ${(error as Error).message}`,
      });
    }
  },

  toggleMic: async () => {
    const { micState } = get();

    if (micState === "recording") {
      const active = recorder;
      recorder = null;
      if (!active) {
        set({ micState: "idle" });
        return;
      }

      set({ micState: "transcribing" });
      try {
        const blob = await active.stop();
        if (blob.size < 1200) {
          set({ micState: "idle", micError: "That was too short to transcribe." });
          return;
        }
        const { text } = await api.transcribeAudio(blob);
        if (!text) {
          set({ micState: "idle", micError: "Nothing was transcribed — try again closer to the mic." });
          return;
        }
        const draft = get().draft;
        set({
          draft: draft ? `${draft.trimEnd()} ${text}` : text,
          micState: "idle",
          micError: null,
        });
      } catch (error) {
        set({ micState: "idle", micError: (error as Error).message });
      }
      return;
    }

    if (micState !== "idle") return;

    try {
      recorder = await startRecording();
      set({ micState: "recording", micError: null });
    } catch (error) {
      set({
        micError: `Microphone unavailable: ${(error as Error).message}`,
      });
    }
  },

  openWorkspace: (view) =>
    set((state) => ({ workspaceView: state.workspaceView === view ? null : view })),

  closeWorkspace: () => set({ workspaceView: null }),

  addPrompt: (title, body) => {
    const next = [{ id: uid(), title: title.trim() || "Untitled", body }, ...get().prompts];
    persistPrompts(next);
    set({ prompts: next });
  },

  removePrompt: (id) => {
    const next = get().prompts.filter((prompt) => prompt.id !== id);
    persistPrompts(next);
    set({ prompts: next });
  },

  importChat: async () => {
    const bridge = window.dizel;
    if (!bridge) return;

    const paths = await bridge.pickFiles("attachment");
    if (paths.length === 0) return;

    const result = await bridge.readTextFile(paths[0]);
    if (!result.ok || typeof result.content !== "string") {
      set({ error: "That file could not be read." });
      return;
    }

    try {
      const parsed = JSON.parse(result.content) as { messages?: unknown; title?: string };
      const raw = Array.isArray(parsed.messages) ? parsed.messages : [];
      const messages = raw
        .filter(
          (item): item is { role?: unknown; content: string } =>
            Boolean(item) &&
            typeof item === "object" &&
            typeof (item as { content?: unknown }).content === "string",
        )
        .map((item) => ({ role: String(item.role ?? "user"), content: item.content }));

      if (messages.length === 0) {
        set({ error: "No messages found in that file." });
        return;
      }

      const title =
        parsed.title ||
        messages.find((m) => m.role === "user")?.content.slice(0, 45) ||
        "Imported chat";

      await api.createSession(messages, title);
      await get().loadSessions();
      set({ error: null });
    } catch (error) {
      set({ error: `Import failed: ${(error as Error).message}` });
    }
  },

  openProfile: () => set({ profileOpen: true }),

  closeProfile: () => set({ profileOpen: false }),

  toggleStatusBar: () => set((state) => ({ statusBarOpen: !state.statusBarOpen })),

  saveProfile: async (username, details, avatar, removeAvatar) => {
    try {
      const result = await api.saveProfile(username, details, avatar, removeAvatar);
      const url = await api.getAvatarUrl();
      const previous = get().avatarUrl;
      if (previous && previous !== url) URL.revokeObjectURL(previous);
      set({
        username: result.username || username || "You",
        profile: profileFromConfig({ ...result }),
        avatarUrl: url,
        profileOpen: false,
        error: null,
      });
    } catch (error) {
      set({ error: (error as Error).message });
    }
  },
}));

function startPolling(set: (partial: Partial<ChatState>) => void): void {
  if (pollTimer !== null) window.clearInterval(pollTimer);

  pollTimer = window.setInterval(async () => {
    try {
      const status = await api.getStatus();
      set({ status });
      if (status.load.status === "ready" || status.load.status === "error") {
        if (pollTimer !== null) window.clearInterval(pollTimer);
        pollTimer = null;
      }
    } catch {
      if (pollTimer !== null) window.clearInterval(pollTimer);
      pollTimer = null;
    }
  }, 800);
}

// Apply the stored theme before first paint.
document.documentElement.dataset.theme = initialTheme();
