/**
 * HTTP client for the local sidecar.
 *
 * Every call carries the launch token issued by the Electron main process, so
 * a stray browser tab on localhost:PORT cannot drive the engine.
 */

import type {
  ChatStatus,
  LoadState,
  ModelEntry,
  ModelFacts,
  ProviderInfo,
  SessionSummary,
  SkillInfo,
  StreamEvent,
} from "../types";
import { resolveSidecar } from "./sidecar";

let base = "";
let token = "";

/** Drop the cached address so the next request re-resolves after a restart. */
export function resetApi(): void {
  base = "";
  token = "";
}

export async function initApi(): Promise<void> {
  const info = await resolveSidecar();
  base = `http://127.0.0.1:${info.port}/api`;
  token = info.token;
}

function authHeaders(extra: Record<string, string> = {}): Record<string, string> {
  return { "Content-Type": "application/json", "X-Dizel-Token": token, ...extra };
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  if (!base) await initApi();

  const response = await fetch(`${base}${path}`, {
    ...init,
    headers: authHeaders(init.headers as Record<string, string> | undefined),
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }

  return (await response.json()) as T;
}

// ── Model / status ──────────────────────────────────────────────────────────

export const getStatus = () => request<ChatStatus>("/chat/status");

export const getLoadStatus = () => request<LoadState>("/chat/load_status");

export const loadModel = (checkpointPath: string, device?: string) =>
  request<LoadState>("/chat/load_model", {
    method: "POST",
    body: JSON.stringify({ checkpoint_path: checkpointPath, device: device ?? null }),
  });

export const stopGeneration = () =>
  request<{ status: string }>("/chat/stop", { method: "POST" });

// ── Sessions ────────────────────────────────────────────────────────────────

export const listSessions = () => request<SessionSummary[]>("/session/list");

export const getSession = (id: string) =>
  request<{ id: string; title: string; messages: { role: string; content: string }[] }>(
    `/session/${encodeURIComponent(id)}`,
  );

export const deleteSession = (id: string) =>
  request<{ status: string }>(`/session/${encodeURIComponent(id)}`, { method: "DELETE" });

export const togglePinSession = (id: string) =>
  request<{ session_id: string; pinned: boolean }>(
    `/session/${encodeURIComponent(id)}/pin`,
    { method: "POST" },
  );

export const archiveSession = (id: string) =>
  request<{ session_id: string; archived: boolean }>(
    `/session/${encodeURIComponent(id)}/archive`,
    { method: "POST" },
  );

export const renameSession = (id: string, title: string) =>
  request<{ session_id: string; title: string }>(
    `/session/${encodeURIComponent(id)}/rename`,
    { method: "POST", body: JSON.stringify({ title }) },
  );

export const truncateSession = (id: string, keep: number) =>
  request<{ session_id: string; kept: number }>(
    `/session/${encodeURIComponent(id)}/truncate`,
    { method: "POST", body: JSON.stringify({ keep }) },
  );

export const createSession = (messages: { role: string; content: string }[], title: string) =>
  request<{ session_id: string }>("/session/", {
    method: "POST",
    body: JSON.stringify({ messages, title, session_id: null }),
  });

export const clearAllSessions = () =>
  request<{ deleted: number }>("/session/clear-all", { method: "POST" });

// ── Feedback ────────────────────────────────────────────────────────────────

export interface FeedbackPayload {
  rating: "up" | "down" | "removed";
  session_id: string;
  message_key: string;
  reasons: string[];
  note: string;
  response: string;
  prompt: string;
  model: string;
  provider: string;
}

export const saveFeedback = (payload: FeedbackPayload) =>
  request<{ status: string; path: string }>("/feedback/", {
    method: "POST",
    body: JSON.stringify(payload),
  });

// ── Skills ─────────────────────────────────────────────────────────────────

export interface SkillCatalog {
  skills: SkillInfo[];
  user_dir: string;
  project_dir: string;
}

export const listSkills = () => request<SkillCatalog>("/skills/");

export const expandSkill = (name: string, args: string) =>
  request<{ name: string; title: string; prompt: string }>("/skills/expand", {
    method: "POST",
    body: JSON.stringify({ name, args }),
  });

// ── Memory ─────────────────────────────────────────────────────────────────

export interface RecentChat {
  id: string;
  title: string;
  summary: string;
  created: string;
}

export const summarizeMessages = (messages: { role: string; content: string }[]) =>
  request<{ summary: string }>("/memory/summarize", {
    method: "POST",
    body: JSON.stringify({ messages }),
  });

export const recentSummaries = (limit = 3) =>
  request<RecentChat[]>(`/memory/recent?limit=${limit}`);

// ── Dictation (Whisper) ────────────────────────────────────────────────────

export interface DictationState {
  state: "idle" | "downloading" | "done" | "error";
  size: string;
  received: number;
  total: number;
  percent: number;
  error: string;
  downloaded: boolean;
}

export const dictationStatus = () =>
  request<{ available: boolean; model: string; downloaded: boolean; reason: string }>(
    "/nova/status",
  );

export const dictationDownload = () => request<DictationState>("/nova/download");

export const startDictationDownload = () =>
  request<DictationState>("/nova/download", { method: "POST" });

export const cancelDictationDownload = () =>
  request<DictationState>("/nova/download/cancel", { method: "POST" });

// ── Context window ─────────────────────────────────────────────────────────

export interface ContextUsage {
  system: number;
  conversation: number;
  reserved: number;
  /** Messages the server dropped to fit the window on the last prompt. */
  trimmed?: number;
  used: number;
  total: number;
  percent: number;
  exact: boolean;
  task: string;
}

export interface UsageWindow {
  used: number;
  capacity: number;
  percent: number;
  exhausted: boolean;
  /** Human countdown, e.g. "4h 36m". */
  resets_in: string;
  compactions_used: number;
  compactions_remaining: number;
  can_compact: boolean;
}

export interface ToolState {
  enabled: boolean;
  available: string[];
}

export interface Capability {
  id: string;
  name: string;
  tui: string | null;
  gui: string | null;
}

export const capabilities = () =>
  request<{ items: Capability[]; table: string | null; report: string }>(
    "/chat/capabilities?table=true",
  );

export const toolsState = () => request<ToolState>("/chat/tools");

export const setTools = (enabled: boolean) =>
  request<ToolState>("/chat/tools", {
    method: "POST",
    body: JSON.stringify({ enabled }),
  });

export const usageWindow = () => request<UsageWindow>("/chat/usage");

export const usageDebug = (action: string, value?: number) =>
  request<UsageWindow & { ok: boolean }>("/chat/usage/debug", {
    method: "POST",
    body: JSON.stringify({ action, value }),
  });

export const contextUsage = (sessionId?: string | null) =>
  request<ContextUsage>(
    `/chat/context${sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : ""}`,
  );

// ── Config ──────────────────────────────────────────────────────────────────

export const getConfig = () => request<Record<string, unknown>>("/config/");

export const saveConfig = (patch: Record<string, unknown>) =>
  request<Record<string, unknown>>("/config/", {
    method: "POST",
    body: JSON.stringify(patch),
  });

export const reloadSettings = () =>
  request<{ status: string }>("/chat/reload_settings", { method: "POST" });

export const resetPreferences = () =>
  request<Record<string, unknown>>("/config/reset_preferences", { method: "POST" });

// ── Models & profiles ───────────────────────────────────────────────────────

export const listModels = () => request<ModelEntry[]>("/models/");

export const listProfiles = () => request<Record<string, string[]>>("/models/profiles");

export const getModelFacts = () => request<ModelFacts>("/models/info");

export const getModelDefaults = () =>
  request<{
    directory: string;
    exists: boolean;
    configured: string;
    device: string;
    cuda_available: boolean;
    sampling: Record<string, number>;
    system_prompt: string;
  }>("/models/defaults");

// ── Providers (BYOK) ────────────────────────────────────────────────────────

export const listProviders = () => request<ProviderInfo[]>("/providers/");

export const setProviderKey = (
  provider: string,
  apiKey: string,
  model: string,
  custom?: { name: string; url: string },
) =>
  request<{ status: string; provider: string; has_key: boolean }>("/providers/key", {
    method: "POST",
    body: JSON.stringify({
      provider,
      api_key: apiKey,
      model,
      custom_name: custom?.name ?? "",
      custom_url: custom?.url ?? "",
    }),
  });

export const listProviderModels = (slug: string) =>
  request<{ models: { id: string; name: string }[] }>(
    `/providers/models?slug=${encodeURIComponent(slug)}`,
  );

export const applyProfile = (variant: string, mode: string) =>
  request<{ status: string; profile: string }>("/chat/apply_profile", {
    method: "POST",
    body: JSON.stringify({ model_name: variant, mode_name: mode }),
  });

// ── Profile ─────────────────────────────────────────────────────────────────

export interface ProfileResult {
  username: string;
  has_avatar: boolean;
  bio: string;
  pronouns: string;
  accent: string;
  greeting_style: string;
  created_at: string;
}

export const getProfile = () => request<ProfileResult>("/profile/");

/** The avatar route needs the launch token, which <img> can't send — so fetch
 *  it and hand back an object URL. */
export async function getAvatarUrl(): Promise<string | null> {
  if (!base) await initApi();

  const response = await fetch(`${base}/profile/avatar`, {
    headers: { "X-Dizel-Token": token },
  });
  if (!response.ok) return null;

  return URL.createObjectURL(await response.blob());
}

export async function saveProfile(
  username: string,
  details: { bio: string; pronouns: string; accent: string; greetingStyle: string },
  avatar?: File,
  removeAvatar = false,
): Promise<ProfileResult> {
  if (!base) await initApi();

  const form = new FormData();
  form.append("username", username);
  form.append("bio", details.bio);
  form.append("pronouns", details.pronouns);
  form.append("accent", details.accent);
  form.append("greeting_style", details.greetingStyle);
  form.append("remove_avatar", removeAvatar ? "true" : "false");
  if (avatar) form.append("avatar", avatar);

  const response = await fetch(`${base}/profile/`, {
    method: "POST",
    headers: { "X-Dizel-Token": token },
    body: form,
  });

  if (!response.ok) {
    let detail = `Could not save profile (${response.status})`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* non-JSON body */
    }
    throw new Error(detail);
  }

  return (await response.json()) as ProfileResult;
}

// ── MCP servers ─────────────────────────────────────────────────────────────

export type McpServerState = "disabled" | "starting" | "ready" | "error";

export interface McpToolInfo {
  name: string;
  description: string;
}

export interface McpServerInfo {
  id: string;
  name: string;
  transport: "stdio" | "http";
  enabled: boolean;
  command?: string;
  args?: string[];
  url?: string;
  env_keys: string[];
  header_keys: string[];
  state: McpServerState;
  error: string;
  tools: McpToolInfo[];
}

export const mcpServers = () =>
  request<{ servers: McpServerInfo[] }>("/mcp/servers");

export const setMcpServers = (servers: unknown[]) =>
  request<{ servers: McpServerInfo[] }>("/mcp/servers", {
    method: "PUT",
    body: JSON.stringify({ servers }),
  });

export const testMcpServer = (id: string) =>
  request<{ server: McpServerInfo | null; state: McpServerState; error: string }>(
    `/mcp/servers/${encodeURIComponent(id)}/test`,
    { method: "POST" },
  );

export interface McpPreferences {
  connect_timeout: number;
  call_timeout: number;
  max_output_chars: number;
  retry_failed_calls: boolean;
  auto_connect: boolean;
}

export const mcpPreferences = () => request<McpPreferences>("/mcp/preferences");

export const setMcpPreferences = (prefs: McpPreferences) =>
  request<McpPreferences>("/mcp/preferences", {
    method: "PUT",
    body: JSON.stringify(prefs),
  });

/** A connected plugin tool, with the schema its arguments follow. */
export interface McpPluginTool {
  name: string;
  /** The name the model sees in its <tool> tag. */
  namespaced: string;
  description: string;
  input_schema: Record<string, unknown>;
}

export interface McpPluginServer {
  id: string;
  name: string;
  enabled: boolean;
  state: McpServerState;
  error: string;
  tools: McpPluginTool[];
}

export interface McpPluginCatalog {
  /** The model's master tools switch; prompt-path calls need it on. */
  tools_enabled: boolean;
  servers: McpPluginServer[];
}

export const mcpTools = () => request<McpPluginCatalog>("/mcp/tools");

export const callMcpTool = (
  serverId: string,
  tool: string,
  args: Record<string, unknown>,
) =>
  request<{
    server: string;
    tool: string;
    namespaced: string;
    result: string;
  }>(
    `/mcp/servers/${encodeURIComponent(serverId)}/tools/${encodeURIComponent(tool)}`,
    { method: "POST", body: JSON.stringify({ arguments: args }) },
  );

// ── Voice ───────────────────────────────────────────────────────────────────

export async function transcribeAudio(blob: Blob): Promise<{ text: string; language?: string }> {
  if (!base) await initApi();

  const form = new FormData();
  form.append("file", blob, "speech.wav");

  const response = await fetch(`${base}/nova/transcribe`, {
    method: "POST",
    // No Content-Type: the browser must set the multipart boundary itself.
    headers: { "X-Dizel-Token": token },
    body: form,
  });

  if (!response.ok) {
    let detail = `Transcription failed (${response.status})`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* non-JSON body */
    }
    throw new Error(detail);
  }

  return (await response.json()) as { text: string; language?: string };
}

// ── Streaming ───────────────────────────────────────────────────────────────

const FRAME_BREAK = /\r?\n\r?\n/;

/**
 * POST /chat/stream and dispatch each SSE frame.
 *
 * Resolves when the stream closes. Throws on transport failure; server-side
 * failures arrive as an `error` event instead.
 */
export async function streamChat(
  userText: string,
  sessionId: string | null,
  onEvent: (event: StreamEvent) => void,
  signal: AbortSignal,
  continuation = false,
  attachments: string[] = [],
): Promise<void> {
  if (!base) await initApi();

  const response = await fetch(`${base}/chat/stream`, {
    method: "POST",
    headers: authHeaders({ Accept: "text/event-stream" }),
    body: JSON.stringify({
      user_text: userText,
      session_id: sessionId,
      attachments,
      continuation,
    }),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(`Stream failed (${response.status} ${response.statusText})`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const dispatch = (frame: string) => {
    const data = frame
      .split(/\r?\n/)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trim())
      .join("\n");

    if (!data) return; // keepalive comment frame
    try {
      onEvent(JSON.parse(data) as StreamEvent);
    } catch {
      /* ignore malformed frame */
    }
  };

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    for (;;) {
      const match = FRAME_BREAK.exec(buffer);
      if (!match) break;
      const frame = buffer.slice(0, match.index);
      buffer = buffer.slice(match.index + match[0].length);
      dispatch(frame);
    }
  }

  if (buffer.trim()) dispatch(buffer);
}
