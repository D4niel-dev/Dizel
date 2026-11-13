export interface SystemStats {
  cpu: number;
  ram: number;
  ramUsedMb: number;
  ramTotalMb: number;
  platform: string;
  uptimeSec: number;
}

export interface ProviderInfo {
  slug: string;
  name: string;
  avatar: string | null;
  requires_key: boolean;
  active: boolean;
  configured: boolean;
  model: string;
}

export interface ModelFacts {
  app_name?: string;
  app_version?: string;
  model_name?: string;
  version?: string;
  model_size?: string;
  context_length?: string;
  vocab_size?: string;
  author?: string;
  description?: string;
  architecture?: string;
  num_layers?: number;
  num_heads?: number;
  kv_heads?: number;
  hidden_dim?: number;
  mlp_size?: number;
  capabilities?: string[];
  training_data?: string;
  train_tokens?: string;
  train_steps?: string;
  repository?: string;
  license?: string;
  build?: string;
  error?: string;
}

export type WorkspaceView =
  | "Chats"
  | "Image"
  | "Presentation"
  | "Riset"
  | "Archived"
  | "Library"
  | "Monitor";

export interface SavedPrompt {
  id: string;
  title: string;
  body: string;
}

export type FontScale = "Small" | "Medium" | "Large";

/** Display preferences that are actually applied by the UI, not just stored. */
export type AccentName = "blue" | "teal" | "violet" | "rose";
export type Density = "comfortable" | "compact";
export type FontFamily = "system" | "serif" | "mono";

export interface Prefs {
  fontScale: FontScale;
  showTimestamps: boolean;
  animations: boolean;
  syntaxHighlight: boolean;
  enterToSend: boolean;
  avatars: boolean;
  accent: AccentName;
  density: Density;
  clock24h: boolean;
  showSpeed: boolean;
  fontFamily: FontFamily;
  /** How wide the transcript column runs. */
  messageWidth: "comfortable" | "wide" | "full";
  /** A hairline between messages. */
  separators: boolean;
  /** Wrap long code lines instead of scrolling them sideways. */
  codeWrap: boolean;
  /** Browser spellcheck in the composer. */
  spellcheck: boolean;
  /** Keep the transcript pinned to the newest message while streaming. */
  autoScroll: boolean;
  /** How long the generating row is held before the answer is shown. */
  thinkingPause: "off" | "short" | "long" | "custom";
  /** Custom hold time in seconds, used when thinkingPause is "custom". */
  thinkingPauseCustom: number;
  /** Gap between turns. */
  messageSpacing: "comfortable" | "compact";
  /** The suggestion pills on an empty chat. */
  showSuggestions: boolean;
  /** Flash the taskbar when a reply lands while the window is in the background. */
  notifyOnFinish: boolean;
  /** Line height of reply and prompt text. */
  lineHeight: "normal" | "relaxed";
  /** Underline links inside replies. */
  underlineLinks: boolean;
  /** Solid surfaces instead of blurred glass. */
  reduceTransparency: boolean;
  /** Corner radius of the user bubble. */
  messageCorners: "rounded" | "soft" | "sharp";
  /** Model name in each reply's meta line. */
  showModelTag: boolean;
  /** How streamed reply text is smoothed as it arrives. */
  typingAnimation: "off" | "fast" | "smooth" | "cinematic";
  /** Focus the composer when a chat opens or the app is ready. */
  autoFocusComposer: boolean;
  /** Escape clears the draft while the composer is focused. */
  escapeClearsDraft: boolean;
  /** Arrow Up in an empty composer edits your last message. */
  arrowUpEditsLast: boolean;
  /** Ctrl+Shift+C copies the latest reply. */
  copyLastReplyHotkey: boolean;
  /** Ctrl+Shift+X stops the current generation. */
  stopGenerationHotkey: boolean;
}

export type ProfileAccent = "blue" | "teal" | "violet" | "rose" | "amber";

/** How the welcome screen greets you. */
export type GreetingStyle = "warm" | "simple" | "time";

export interface ProfileDetails {
  bio: string;
  pronouns: string;
  accent: ProfileAccent;
  greetingStyle: GreetingStyle;
  createdAt: string;
}

export interface SidecarInfo {
  port: number;
  token: string;
}

export type UpdateStatus =
  | "idle"
  | "checking"
  | "available"
  | "downloading"
  | "downloaded"
  | "up-to-date"
  | "error";

export interface UpdateState {
  status: UpdateStatus;
  currentVersion: string;
  latestVersion: string;
  notes: string;
  releaseUrl: string;
  progress: number;
  error: string;
  autoCheck: boolean;
  canInstall: boolean;
  dev: boolean;
}

export interface AttachmentFile {
  ok: boolean;
  content?: string;
  size?: number;
  truncated?: boolean;
  error?: string;
}

export interface Attachment {
  id: string;
  name: string;
  path: string;
  size: number;
  content: string;
  truncated: boolean;
  /** Data URL for image attachments — shown to you, not sent to the model. */
  previewUrl?: string;
}

export interface ModelEntry {
  name: string;
  stem: string;
  display_name: string;
  path: string;
  size: number;
  modified: number;
  kind: string;
}

export interface AppConfig {
  checkpoint?: string;
  device?: string;
  system_prompt?: string;
  sampling?: {
    temperature?: number;
    top_k?: number;
    top_p?: number;
    repetition_penalty?: number;
    max_new_tokens?: number;
  };
  appearance?: { theme?: string };
  user_profile?: { username?: string };
  [key: string]: unknown;
}

export type Role = "user" | "assistant" | "system";

export type FeedbackRating = "up" | "down";

export interface MessageFeedback {
  rating: FeedbackRating;
  reasons: string[];
  note: string;
  createdAt: number;
}

/** One version of a turn: the prompt as edited, and the reply it produced. */
export interface Fork {
  prompt: string;
  reply: Message | null;
}

export interface Message {
  id: string;
  role: Role;
  content: string;
  createdAt: number;
  streaming?: boolean;
  failed?: boolean;
  durationMs?: number;
  /** Reasoning streamed before the answer, when the model exposes one. */
  thinking?: string;
  /** Wall-clock start of the reasoning phase, for the live Thinking clock. */
  thoughtStartedAt?: number;
  /** Time spent in that reasoning phase before the first answer token. */
  thoughtMs?: number;
  /** Versions of this turn. On a user message, switching swaps the prompt AND
   *  the reply below it — that is what editing a question actually forks. */
  forks?: Fork[];
  forkIndex?: number;
  /** Other replies to the same prompt. When present, the bubble shows a ‹ ›
   *  switcher and renders `siblings[siblingIndex]` instead of its own content. */
  siblings?: Message[];
  siblingIndex?: number;
  /** How long this reply was held behind the generating row. Set at send time
   *  so the pause can scale with the prompt without the bubble knowing it. */
  holdMs?: number;
  /** Files this message carried. Their text is also folded into `content` for
   *  the model; this copy exists so the transcript can show and open them. */
  attachments?: Attachment[];
}

export interface ModelInfo {
  provider?: string;
  params?: string;
  device?: string;
  ctx_len?: number;
  /** True when no model reported its window and the value came from settings. */
  ctx_assumed?: boolean;
  /** Whether the loaded model can accept image input. */
  vision?: boolean;
  d_model?: number;
  n_layers?: number;
  n_heads?: number;
  vocab?: number;
}

export type LoadStatus = "idle" | "loading" | "ready" | "error";

export interface LoadState {
  status: LoadStatus;
  message: string;
  error: string;
  checkpoint: string;
  display_name?: string;
  is_ready?: boolean;
}

export interface ChatStatus {
  is_ready: boolean;
  is_generating: boolean;
  provider: string;
  api_model: string;
  model_info: ModelInfo;
  load: LoadState;
}

export interface SessionSummary {
  id: string;
  title: string;
  created: string;
  preview: string;
  pinned: boolean;
  archived: boolean;
  tags: string[];
}

export interface SkillInfo {
  name: string;
  title: string;
  description: string;
  aliases: string[];
  category: string;
  source: string;
  builtin: boolean;
}

export type StreamEvent =
  | { type: "start"; session_id: string }
  | { type: "thinking"; text: string }
  | { type: "token"; text: string }
  | { type: "notice"; text: string }
  | { type: "done"; text: string; session_id: string }
  | { type: "error"; message: string };
