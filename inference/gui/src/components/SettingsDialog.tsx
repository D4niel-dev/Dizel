import { useEffect, useRef, useState } from "react";

import { Icon } from "./Icon";
import type { IconName } from "./Icon";
import { Logo } from "./Logo";
import { McpSettings } from "./McpSettings";
import { ProviderAvatar } from "./ProviderAvatar";
import { Segmented } from "./Segmented";
import { Row, Section, Switch } from "./settingsUi";
import { useUpdates } from "../hooks/useUpdates";
import * as api from "../services/api";
import { prefsFromConfig, useChat } from "../store/chatStore";
import type { ThemeName } from "../store/chatStore";
import type {
  AccentName,
  AppConfig,
  Density,
  FontFamily,
  FontScale,
  ModelFacts,
  ProviderInfo,
} from "../types";

const TABS = ["Chat", "MCP", "Model", "Appearance", "Speech", "Shortcuts", "About"] as const;

const TAB_ICONS: Record<(typeof TABS)[number], IconName> = {
  Chat: "chat",
  MCP: "plug",
  Model: "sparkles",
  Appearance: "image",
  Speech: "mic",
  Shortcuts: "keyboard",
  About: "info",
};
type Tab = (typeof TABS)[number];

const TAB_HINTS: Record<Tab, string> = {
  Chat: "Checkpoint, provider, prompt and sampling",
  MCP: "Tool servers, timeouts and limits",
  Model: "Response budgets and what this build actually is",
  Appearance: "Theme and how messages are displayed",
  Speech: "Dictation model, language and decoding",
  Shortcuts: "Keyboard bindings and composer habits",
  About: "Version, backups, links and your data",
};

const THEMES: { value: ThemeName; label: string }[] = [
  { value: "dark", label: "Dark" },
  { value: "light", label: "Light" },
  { value: "dark-blue", label: "Dark blue" },
];

const FONT_SCALES: { value: FontScale; label: string }[] = [
  { value: "Small", label: "Small" },
  { value: "Medium", label: "Medium" },
  { value: "Large", label: "Large" },
];

interface Sampling {
  temperature: number;
  top_k: number;
  top_p: number;
  repetition_penalty: number;
  max_new_tokens: number;
}

const DEFAULT_SAMPLING: Sampling = {
  temperature: 0.7,
  top_k: 60,
  top_p: 0.9,
  repetition_penalty: 1.2,
  max_new_tokens: 512,
};

interface TokenBudget {
  chat_budget: number;
  coding_budget: number;
  complex_budget: number;
  factual_budget: number;
  tool_budget: number;
  max_context_tokens: number;
  verbosity: string;
}

const DEFAULT_BUDGET: TokenBudget = {
  chat_budget: 150,
  coding_budget: 350,
  complex_budget: 500,
  factual_budget: 100,
  tool_budget: 300,
  max_context_tokens: 10000,
  verbosity: "normal",
};

const DEFAULT_APPEARANCE = {
  theme: "dark" as ThemeName,
  fontScale: "Medium" as FontScale,
  showTimestamps: true,
  animations: true,
  syntaxHighlight: true,
  enterToSend: true,
  avatars: true,
  accent: "blue" as AccentName,
  density: "comfortable" as Density,
  clock24h: true,
  showSpeed: true,
  fontFamily: "system" as FontFamily,
};

function customPauseSeconds(value: string): number {
  const seconds = Number(value);
  return Number.isFinite(seconds)
    ? Math.min(86400, Math.max(1, Math.round(seconds)))
    : 60;
}

function ToggleRow({
  label,
  hint,
  checked,
  onChange,
}: {
  label: string;
  hint?: string;
  checked: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <button
      className="set-row as-button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
    >
      <span className="set-row-text">
        <span>{label}</span>
        {hint && <small>{hint}</small>}
      </span>
      <span className={checked ? "switch on" : "switch"}>
        <span className="switch-knob" />
      </span>
    </button>
  );
}

function Select({
  value,
  onChange,
  options,
  disabled,
}: {
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string; disabled?: boolean }[];
  disabled?: boolean;
}) {
  return (
    <select
      className="set-select"
      value={value}
      disabled={disabled}
      onChange={(event) => onChange(event.target.value)}
    >
      {options.map((option) => (
        <option key={option.value} value={option.value} disabled={option.disabled}>
          {option.label}
        </option>
      ))}
    </select>
  );
}

export function SettingsDialog() {
  const open = useChat((s) => s.settingsOpen);
  const settingsTab = useChat((s) => s.settingsTab);
  const close = useChat((s) => s.closeSettings);
  const save = useChat((s) => s.saveSettings);
  const models = useChat((s) => s.models);
  const loadSessions = useChat((s) => s.loadSessions);
  const currentTheme = useChat((s) => s.theme);
  const settingsError = useChat((s) => s.settingsError);
  const prefs = useChat((s) => s.prefs);
  const sessionCount = useChat((s) => s.sessions.length);
  const runningDevice = useChat((s) => s.status?.model_info?.device);
  const refreshStatus = useChat((s) => s.refreshStatus);

  const [tab, setTab] = useState<Tab>("Chat");
  const [loading, setLoading] = useState(true);
  const [notice, setNotice] = useState<string | null>(null);
  const [confirmClear, setConfirmClear] = useState(false);
  const [confirmCache, setConfirmCache] = useState(false);
  const [confirmReset, setConfirmReset] = useState(false);

  // Chat
  const [checkpoint, setCheckpoint] = useState("");
  const [device, setDevice] = useState("cpu");
  const [cudaAvailable, setCudaAvailable] = useState(false);
  const [ggufFormat, setGgufFormat] = useState("auto");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [sampling, setSampling] = useState<Sampling>(DEFAULT_SAMPLING);
  const [navOpen, setNavOpen] = useState(true);
  const [scrolled, setScrolled] = useState(false);
  const [tools, setTools] = useState<api.ToolState | null>(null);
  const [usage, setUsage] = useState<api.UsageWindow | null>(null);
  const [budget, setBudget] = useState<TokenBudget>(DEFAULT_BUDGET);
  const [hardCap, setHardCap] = useState("0");
  const [personas, setPersonas] = useState<Record<string, string>>({});
  const [ollamaUrl, setOllamaUrl] = useState("http://localhost:11434");

  // Providers
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [activeProvider, setActiveProvider] = useState("local");
  const [apiKey, setApiKey] = useState("");
  const [apiModel, setApiModel] = useState("");
  const [keyStatus, setKeyStatus] = useState<string | null>(null);
  const [ollamaModels, setOllamaModels] = useState<{ id: string; name: string }[]>([]);
  const [ollamaError, setOllamaError] = useState<string | null>(null);
  const [customName, setCustomName] = useState("");
  const [customUrl, setCustomUrl] = useState("");
  const [customModels, setCustomModels] = useState<{ id: string; name: string }[]>([]);
  const [customError, setCustomError] = useState<string | null>(null);
  const [loadingModels, setLoadingModels] = useState(false);
  const [activating, setActivating] = useState(false);

  // Appearance
  const [theme, setTheme] = useState<ThemeName>(currentTheme);
  const [fontScale, setFontScale] = useState<FontScale>("Medium");
  const [showTimestamps, setShowTimestamps] = useState(true);
  const [animations, setAnimations] = useState(true);
  const [syntaxHighlight, setSyntaxHighlight] = useState(true);
  const [enterToSend, setEnterToSend] = useState(true);
  const [avatars, setAvatars] = useState(true);
  const [thinkingPause, setThinkingPause] = useState<
    "off" | "short" | "long" | "custom"
  >("short");
  const [customPause, setCustomPause] = useState("60");
  const [accent, setAccent] = useState<AccentName>("blue");
  const [density, setDensity] = useState<Density>("comfortable");
  const [clock24h, setClock24h] = useState(true);
  const [showSpeed, setShowSpeed] = useState(true);
  const [fontFamily, setFontFamily] = useState<FontFamily>("system");
  const [messageWidth, setMessageWidth] = useState<"comfortable" | "wide" | "full">(
    "comfortable",
  );
  const [separators, setSeparators] = useState(false);
  const [codeWrap, setCodeWrap] = useState(false);
  const [spellcheck, setSpellcheck] = useState(true);
  const [autoScroll, setAutoScroll] = useState(true);
  const [messageSpacing, setMessageSpacing] = useState<"comfortable" | "compact">(
    "comfortable",
  );
  const [showSuggestions, setShowSuggestions] = useState(true);
  const [notifyOnFinish, setNotifyOnFinish] = useState(true);
  const [lineHeight, setLineHeight] = useState<"normal" | "relaxed">("normal");
  const [underlineLinks, setUnderlineLinks] = useState(true);
  const [reduceTransparency, setReduceTransparency] = useState(false);
  const [messageCorners, setMessageCorners] = useState<"rounded" | "soft" | "sharp">(
    "rounded",
  );
  const [showModelTag, setShowModelTag] = useState(false);
  const [typingAnimation, setTypingAnimation] = useState<
    "off" | "fast" | "smooth" | "cinematic"
  >("smooth");
  const [autoFocusComposer, setAutoFocusComposer] = useState(true);
  const [escapeClearsDraft, setEscapeClearsDraft] = useState(false);
  const [arrowUpEditsLast, setArrowUpEditsLast] = useState(true);
  const [copyLastReplyHotkey, setCopyLastReplyHotkey] = useState(true);
  const [stopGenerationHotkey, setStopGenerationHotkey] = useState(true);
  const toggleDebug = useChat((s) => s.toggleDebug);
  const [ctxLength, setCtxLength] = useState("2048");
  const [routerBlock, setRouterBlock] = useState<Record<string, unknown>>({});
  const familyClicks = useRef(0);
  const familyTimer = useRef<number | null>(null);

  // Hidden unlock: six clicks on the model's Family value toggles debug mode.
  const handleFamilyClick = () => {
    familyClicks.current += 1;
    if (familyTimer.current) window.clearTimeout(familyTimer.current);
    familyTimer.current = window.setTimeout(() => {
      familyClicks.current = 0;
    }, 1500);

    if (familyClicks.current >= 6) {
      familyClicks.current = 0;
      toggleDebug();
    }
  };

  // Speech
  const [whisperSize, setWhisperSize] = useState("base");
  const [language, setLanguage] = useState("auto");
  const [whisperTask, setWhisperTask] = useState("transcribe");
  const [whisperPrompt, setWhisperPrompt] = useState("");
  const [beamSize, setBeamSize] = useState(5);
  const [bestOf, setBestOf] = useState(5);
  const [whisperTemp, setWhisperTemp] = useState(0);
  const [conditionPrev, setConditionPrev] = useState(true);
  const [noSpeechThreshold, setNoSpeechThreshold] = useState(0.6);

  const [facts, setFacts] = useState<ModelFacts | null>(null);
  const update = useUpdates();
  const appVersion = update?.currentVersion || facts?.app_version;
  const updateBusy = update?.status === "checking" || update?.status === "downloading";

  let updateTitle = "GitHub Releases";
  let updateMark: IconName = "refresh";
  let updateText = "Checks run automatically in the background.";
  if (!update) {
    updateTitle = "Updates";
    updateMark = "info";
    updateText = "Update controls are available in the desktop app.";
  } else if (update.status === "checking") {
    updateTitle = "Checking for updates";
    updateText = "Checking GitHub Releases…";
  } else if (update.status === "available") {
    updateTitle = `Dizel ${update.latestVersion}`;
    updateMark = "download";
    updateText = "A newer stable build is available.";
  } else if (update.status === "downloading") {
    updateTitle = `Dizel ${update.latestVersion}`;
    updateMark = "download";
    updateText = `Downloading — ${update.progress}%`;
  } else if (update.status === "downloaded") {
    updateTitle = `Dizel ${update.latestVersion} is ready`;
    updateMark = "check";
    updateText = "Restart to finish installing.";
  } else if (update.status === "up-to-date") {
    updateTitle = "Up to date";
    updateMark = "check";
    updateText = `Dizel ${update.currentVersion} is the latest stable build.`;
  } else if (update.status === "error") {
    updateTitle = "Update check failed";
    updateMark = "alert";
    updateText = update.error || "The update check failed.";
  }

  useEffect(() => {
    if (!open) return;
    const target = (TABS as readonly string[]).includes(settingsTab)
      ? (settingsTab as Tab)
      : "Chat";
    setTab(target);
  }, [open, settingsTab]);

  useEffect(() => {
    if (!open) return;

    let cancelled = false;
    setLoading(true);
    setNotice(null);
    setConfirmClear(false);
    setConfirmCache(false);
    setConfirmReset(false);

    Promise.all([
      api.getConfig(),
      api.listProviders(),
      api.getModelFacts().catch(() => null),
      api.getModelDefaults().catch(() => null),
    ])
      .then(([config, providerList, modelFacts, defaults]) => {
        if (cancelled) return;
        const cfg = config as AppConfig;

        setCheckpoint(typeof cfg.checkpoint === "string" ? cfg.checkpoint : "");
        setDevice(typeof cfg.device === "string" ? cfg.device : "cpu");
        setCudaAvailable(Boolean(defaults?.cuda_available));
        setSystemPrompt(typeof cfg.system_prompt === "string" ? cfg.system_prompt : "");
        setSampling({ ...DEFAULT_SAMPLING, ...(cfg.sampling ?? {}) });
        void api.toolsState().then(setTools).catch(() => undefined);
        void api.usageWindow().then(setUsage).catch(() => undefined);

        const advanced = (cfg as Record<string, unknown>).advanced as
          | { gguf_chat_format?: string }
          | undefined;
        setGgufFormat(advanced?.gguf_chat_format ?? "auto");

        const budgetCfg = ((cfg as Record<string, unknown>).token_budget ?? {}) as Record<
          string,
          unknown
        >;
        setBudget({
          ...DEFAULT_BUDGET,
          ...(budgetCfg as Partial<TokenBudget>),
          verbosity:
            budgetCfg.verbosity === "low" || budgetCfg.verbosity === "high"
              ? (budgetCfg.verbosity as string)
              : "normal",
        });
        setHardCap(String(Number(budgetCfg.hard_output_limit) || 0));
        const personaMap = (cfg as Record<string, unknown>).system_prompt_profiles;
        setPersonas(
          personaMap && typeof personaMap === "object"
            ? (personaMap as Record<string, string>)
            : {},
        );

        // Read appearance through the values the app is actually running with,
        // so the dialog can't drift from reality.
        setTheme(currentTheme);
        setFontScale(prefs.fontScale);
        setShowTimestamps(prefs.showTimestamps);
        setAnimations(prefs.animations);
        setSyntaxHighlight(prefs.syntaxHighlight);
        setEnterToSend(prefs.enterToSend);
        setAvatars(prefs.avatars);
        setThinkingPause(prefs.thinkingPause);
        setCustomPause(String(prefs.thinkingPauseCustom));
        setAccent(prefs.accent);
        setDensity(prefs.density);
        setClock24h(prefs.clock24h);
        setShowSpeed(prefs.showSpeed);
        setFontFamily(prefs.fontFamily);
        setMessageWidth(prefs.messageWidth);
        setSeparators(prefs.separators);
        setCodeWrap(prefs.codeWrap);
        setSpellcheck(prefs.spellcheck);
        setAutoScroll(prefs.autoScroll);
        setMessageSpacing(prefs.messageSpacing);
        setShowSuggestions(prefs.showSuggestions);
        setNotifyOnFinish(prefs.notifyOnFinish);
        setLineHeight(prefs.lineHeight);
        setUnderlineLinks(prefs.underlineLinks);
        setReduceTransparency(prefs.reduceTransparency);
        setMessageCorners(prefs.messageCorners);
        setShowModelTag(prefs.showModelTag);
        setTypingAnimation(prefs.typingAnimation);
        setAutoFocusComposer(prefs.autoFocusComposer);
        setEscapeClearsDraft(prefs.escapeClearsDraft);
        setArrowUpEditsLast(prefs.arrowUpEditsLast);
        setCopyLastReplyHotkey(prefs.copyLastReplyHotkey);
        setStopGenerationHotkey(prefs.stopGenerationHotkey);

        const nova = ((cfg as Record<string, unknown>).nova ?? {}) as Record<string, unknown>;
        setWhisperSize((nova.model_size as string) ?? "base");
        setLanguage((nova.language as string) ?? "auto");
        setWhisperTask((nova.task as string) ?? "transcribe");
        setWhisperPrompt((nova.initial_prompt as string) ?? "");
        setBeamSize(Number(nova.beam_size) || 5);
        setBestOf(Number(nova.best_of) || 5);
        setWhisperTemp(Number.isFinite(Number(nova.temperature)) ? Number(nova.temperature) : 0);
        setConditionPrev(nova.condition_on_previous_text !== false);
        setNoSpeechThreshold(
          Number.isFinite(Number(nova.no_speech_threshold)) ? Number(nova.no_speech_threshold) : 0.6,
        );

        setProviders(providerList);
        const active = providerList.find((provider) => provider.active);
        setActiveProvider(active?.slug ?? "local");
        setApiModel(active?.model ?? "");
        setApiKey("");
        setKeyStatus(null);

        const router = ((cfg as Record<string, unknown>).api_router ?? {}) as Record<string, unknown>;
        setRouterBlock(router);
        setCtxLength(String(router.context_length ?? 2048));
        setOllamaUrl((router.ollama_url as string) || "http://localhost:11434");
        setCustomName((router.custom_name as string) ?? "");
        setCustomUrl((router.custom_url as string) ?? "");
        setFacts(modelFacts);

        setLoading(false);
      })
      .catch(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [open, currentTheme, prefs]);

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") close();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, close]);

  if (!open) return null;

  const browse = async () => {
    const paths = await window.dizel?.pickFiles("checkpoint");
    if (paths && paths.length > 0) setCheckpoint(paths[0]);
  };

  const submit = () => {
    void save({
      checkpoint,
      device,
      system_prompt: systemPrompt,
      sampling,
      token_budget: { ...budget, hard_output_limit: Math.max(0, Number(hardCap) || 0) },
      advanced: { gguf_chat_format: ggufFormat },
      appearance: {
        theme,
        color_mode: theme,
        font_scale: fontScale,
        show_timestamps: showTimestamps,
        animations,
        syntax_highlighting: syntaxHighlight,
        enter_to_send: enterToSend,
        avatar_display: avatars ? "Show avatars" : "Hide avatars",
        thinking_pause:
          thinkingPause === "off"
            ? "Off"
            : thinkingPause === "long"
              ? "Long (10s)"
              : thinkingPause === "custom"
                ? "Custom"
                : "Short",
        thinking_pause_custom: customPauseSeconds(customPause),
        accent,
        density,
        clock_24h: clock24h,
        show_speed: showSpeed,
        font_family: fontFamily,
        message_width: messageWidth,
        separators,
        code_wrap: codeWrap,
        spellcheck,
        auto_scroll: autoScroll,
        message_spacing: messageSpacing,
        show_suggestions: showSuggestions,
        notify_on_finish: notifyOnFinish,
        line_height: lineHeight,
        underline_links: underlineLinks,
        reduce_transparency: reduceTransparency,
        message_corners: messageCorners,
        show_model_tag: showModelTag,
        typing_animation: typingAnimation,
        auto_focus_composer: autoFocusComposer,
        escape_clears_draft: escapeClearsDraft,
        arrow_up_edits_last: arrowUpEditsLast,
        copy_last_reply_hotkey: copyLastReplyHotkey,
        stop_generation_hotkey: stopGenerationHotkey,
      },
      api_router: {
        ...routerBlock,
        context_length: Number(ctxLength) || 2048,
        ollama_url: ollamaUrl.trim() || "http://localhost:11434",
      },
      nova: {
        model_size: whisperSize,
        language,
        task: whisperTask,
        initial_prompt: whisperPrompt,
        beam_size: beamSize,
        best_of: Math.max(beamSize, bestOf),
        temperature: whisperTemp,
        condition_on_previous_text: conditionPrev,
        no_speech_threshold: noSpeechThreshold,
      },
    });
  };

  const clearChats = async () => {
    try {
      const result = await api.clearAllSessions();
      await loadSessions();
      setConfirmClear(false);
      setNotice(`Deleted ${result.deleted} chat${result.deleted === 1 ? "" : "s"}.`);
    } catch (error) {
      setNotice((error as Error).message);
    }
  };

  const exportSettings = async () => {
    try {
      // getConfig redacts MCP env/headers, so the export carries no secrets.
      const cfg = await api.getConfig();
      const result = await window.dizel?.saveJson(
        "dizel-settings.json",
        JSON.stringify(cfg, null, 2),
      );
      if (result && typeof result === "object" && "error" in result) {
        setNotice(result.error);
      } else if (result) {
        setNotice("Settings exported.");
      }
    } catch (error) {
      setNotice((error as Error).message);
    }
  };

  const importSettings = async () => {
    const bridge = window.dizel;
    if (!bridge) return;
    try {
      const paths = await bridge.pickFiles("attachment");
      if (paths.length === 0) return;
      const file = await bridge.readTextFile(paths[0]);
      if (!file.ok || typeof file.content !== "string") {
        setNotice("That file could not be read.");
        return;
      }
      const parsed = JSON.parse(file.content) as unknown;
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        setNotice("That file is not a Dizel settings export.");
        return;
      }
      // saveSettings persists, reloads the sidecar, applies appearance and
      // closes this dialog; the toast confirms the import.
      await save(parsed as Record<string, unknown>);
      useChat.getState().notify("Settings imported.");
    } catch (error) {
      setNotice(`Import failed: ${(error as Error).message}`);
    }
  };

  const copyDiagnostics = async () => {
    const state = useChat.getState();
    const bundle = {
      app: appVersion ?? null,
      build: facts?.build ?? null,
      checkpoint,
      device,
      running_device: runningDevice ?? null,
      ready: state.status?.is_ready ?? false,
      provider: state.status?.provider ?? "local",
      model: state.status?.api_model ?? "",
      theme: currentTheme,
      saved_chats: sessionCount,
      sidecar: state.sidecarDown ? "down" : "up",
      platform: navigator.platform,
      user_agent: navigator.userAgent,
      prefs,
    };
    try {
      await navigator.clipboard.writeText(JSON.stringify(bundle, null, 2));
      setNotice("Diagnostics copied to the clipboard.");
    } catch {
      setNotice("Clipboard unavailable.");
    }
  };

  const clearUiCache = () => {
    try {
      localStorage.clear();
    } catch {
      /* storage may be unavailable */
    }
    window.location.reload();
  };

  const resetPreferences = async () => {
    setConfirmReset(false);
    try {
      await api.resetPreferences();
      await api.reloadSettings();
      const cfg = await api.getConfig();
      const appearance = (cfg.appearance ?? {}) as Record<string, unknown>;
      useChat.getState().setPrefs(prefsFromConfig(appearance));
      const theme = appearance.theme;
      if (theme === "dark" || theme === "light" || theme === "dark-blue") {
        useChat.getState().setTheme(theme);
      }
      await refreshStatus();
      setNotice("Preferences reset to defaults.");
    } catch (error) {
      setNotice((error as Error).message);
    }
  };

  const resetSection = () => {
    if (tab === "Chat") {
      setSampling(DEFAULT_SAMPLING);
      setSystemPrompt("");
      setGgufFormat("auto");
      setBudget((prev) => ({
        ...prev,
        max_context_tokens: DEFAULT_BUDGET.max_context_tokens,
        verbosity: DEFAULT_BUDGET.verbosity,
      }));
      setHardCap("0");
      setOllamaUrl("http://localhost:11434");
    }
    if (tab === "Model") {
      setBudget((prev) => ({
        ...prev,
        chat_budget: DEFAULT_BUDGET.chat_budget,
        coding_budget: DEFAULT_BUDGET.coding_budget,
        complex_budget: DEFAULT_BUDGET.complex_budget,
        factual_budget: DEFAULT_BUDGET.factual_budget,
        tool_budget: DEFAULT_BUDGET.tool_budget,
      }));
    }
    if (tab === "Appearance") {
      setTheme(DEFAULT_APPEARANCE.theme);
      setFontScale(DEFAULT_APPEARANCE.fontScale);
      setShowTimestamps(DEFAULT_APPEARANCE.showTimestamps);
      setAnimations(DEFAULT_APPEARANCE.animations);
      setSyntaxHighlight(DEFAULT_APPEARANCE.syntaxHighlight);
      setEnterToSend(DEFAULT_APPEARANCE.enterToSend);
      setAvatars(DEFAULT_APPEARANCE.avatars);
      setThinkingPause("short");
      setCustomPause("60");
      setAccent(DEFAULT_APPEARANCE.accent);
      setDensity(DEFAULT_APPEARANCE.density);
      setClock24h(DEFAULT_APPEARANCE.clock24h);
      setShowSpeed(DEFAULT_APPEARANCE.showSpeed);
      setFontFamily(DEFAULT_APPEARANCE.fontFamily);
      setMessageWidth("comfortable");
      setSeparators(false);
      setCodeWrap(false);
      setSpellcheck(true);
      setAutoScroll(true);
      setMessageSpacing("comfortable");
      setShowSuggestions(true);
      setNotifyOnFinish(true);
      setLineHeight("normal");
      setUnderlineLinks(true);
      setReduceTransparency(false);
      setMessageCorners("rounded");
      setShowModelTag(false);
      setTypingAnimation("smooth");
    }
    if (tab === "Speech") {
      setWhisperSize("base");
      setLanguage("auto");
      setWhisperTask("transcribe");
      setWhisperPrompt("");
      setBeamSize(5);
      setBestOf(5);
      setWhisperTemp(0);
      setConditionPrev(true);
      setNoSpeechThreshold(0.6);
    }
    if (tab === "Shortcuts") {
      setAutoFocusComposer(true);
      setEscapeClearsDraft(false);
      setArrowUpEditsLast(true);
      setCopyLastReplyHotkey(true);
      setStopGenerationHotkey(true);
    }
  };

  const selected = providers.find((provider) => provider.slug === activeProvider);
  const gridProviders = providers.filter((provider) => provider.slug !== "custom");
  const customProvider = providers.find((provider) => provider.slug === "custom");

  const chooseProvider = (provider: ProviderInfo) => {
    setActiveProvider(provider.slug);
    setApiModel(provider.model);
    setApiKey("");
    setKeyStatus(null);
    setOllamaError(null);

    if (provider.slug === "ollama") void loadOllamaModels(provider.model);
    // Local needs no configuration, so selecting it can switch straight away.
    if (provider.slug === "local") void activate("local", "");
  };

  const activate = async (
    slug: string,
    model: string,
    key = "",
    custom?: { name: string; url: string },
  ) => {
    setActivating(true);
    setKeyStatus(null);
    try {
      await api.setProviderKey(slug, key, model, custom);
      setProviders(await api.listProviders());
      setKeyStatus("Active.");
      await refreshStatus();
    } catch (error) {
      setKeyStatus((error as Error).message);
    } finally {
      setActivating(false);
    }
  };

  /**
   * Custom providers are built from a name + base URL, so they can't be
   * activated until those exist. Save them, then pull the model list from the
   * endpoint and activate with the first model.
   */
  const connectCustom = async () => {
    setActivating(true);
    setKeyStatus(null);
    setCustomError(null);
    const custom = { name: customName.trim() || "Custom Provider", url: customUrl.trim() };

    try {
      await api.setProviderKey("custom", apiKey, apiModel, custom);
      const { models } = await api.listProviderModels("custom");
      setCustomModels(models);

      if (models.length > 0) {
        setApiModel(models[0].id);
        await api.setProviderKey("custom", apiKey, models[0].id, custom);
        setKeyStatus(`Connected · ${models.length} model${models.length === 1 ? "" : "s"}`);
      } else {
        setKeyStatus("Connected, but the endpoint listed no models.");
      }

      setProviders(await api.listProviders());
      await refreshStatus();
    } catch (error) {
      setCustomModels([]);
      setCustomError((error as Error).message);
    } finally {
      setActivating(false);
    }
  };

  const loadOllamaModels = async (preferred = "") => {
    setLoadingModels(true);
    setOllamaError(null);
    try {
      const { models: found } = await api.listProviderModels("ollama");
      setOllamaModels(found);
      if (found.length === 0) {
        setOllamaError("Ollama is running but has no models — try `ollama pull llama3.2`.");
      } else if (!found.some((model) => model.id === preferred)) {
        setApiModel(found[0].id);
      }
    } catch (error) {
      setOllamaModels([]);
      setOllamaError((error as Error).message);
    } finally {
      setLoadingModels(false);
    }
  };

  const applyOllamaUrl = async () => {
    const url = ollamaUrl.trim() || "http://localhost:11434";
    setActivating(true);
    setKeyStatus(null);
    try {
      const next = { ...routerBlock, ollama_url: url };
      await api.saveConfig({ api_router: next });
      await api.reloadSettings();
      setRouterBlock(next);
      setOllamaUrl(url);
      await loadOllamaModels(apiModel);
      setKeyStatus("Server URL saved.");
    } catch (error) {
      setKeyStatus((error as Error).message);
    } finally {
      setActivating(false);
    }
  };

  const providerCard = (provider: ProviderInfo, wide = false) => (
    <button
      key={provider.slug}
      className={[
        "provider-card",
        wide ? "wide" : "",
        provider.slug === activeProvider ? "active" : "",
      ]
        .filter(Boolean)
        .join(" ")}
      onClick={() => chooseProvider(provider)}
      title={provider.name}
    >
      <ProviderAvatar
        avatar={provider.avatar}
        logo={provider.slug === "local"}
        size={wide ? 18 : 26}
      />
      <span>{provider.name}</span>
    </button>
  );

  const slider = (
    key: keyof Sampling,
    label: string,
    min: number,
    max: number,
    step: number,
    format: (value: number) => string,
  ) => (
    <label className="set-row" key={key}>
      <span className="set-row-text">
        <span>{label}</span>
      </span>
      <span className="set-row-control slider-control">
        <b>{format(sampling[key])}</b>
        <input
          className="set-range"
          type="range"
          min={min}
          max={max}
          step={step}
          value={sampling[key]}
          onChange={(event) =>
            setSampling((prev) => ({ ...prev, [key]: Number(event.target.value) }))
          }
        />
      </span>
    </label>
  );

  const rangeRow = (
    key: string,
    label: string,
    hint: string,
    value: number,
    min: number,
    max: number,
    step: number,
    format: (value: number) => string,
    onChange: (value: number) => void,
  ) => (
    <label className="set-row" key={key}>
      <span className="set-row-text">
        <span>{label}</span>
        <small>{hint}</small>
      </span>
      <span className="set-row-control slider-control">
        <b>{format(value)}</b>
        <input
          className="set-range"
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(event) => onChange(Number(event.target.value))}
        />
      </span>
    </label>
  );

  return (
    <div
      className="modal-backdrop"
      onMouseDown={(event) => event.target === event.currentTarget && close()}
    >
      <div className="modal settings-modal" role="dialog" aria-modal="true" aria-label="Settings">
        <header className="set-titlebar">
          <div className="set-titlebar-side">
            <button
              className="set-nav-toggle"
              onClick={() => setNavOpen((open) => !open)}
              aria-label={navOpen ? "Hide the section list" : "Show the section list"}
              title={navOpen ? "Hide sections" : "Show sections"}
            >
              <Icon name={navOpen ? "chevronLeft" : "chevronRight"} size={14} />
            </button>
          </div>

          {/* Centred on the modal itself, not on the space left over after the
              buttons — that is why this is a grid rather than a flex row. */}
          <div className="set-titlebar-center">
            <Logo size={18} />
            <span className="set-title">Settings</span>
          </div>

          <div className="set-titlebar-side is-right">
            <button className="modal-x" onClick={close} aria-label="Close settings">
              <Icon name="close" size={13} />
            </button>
          </div>
        </header>

        <div className="set-shell">
          <nav className={navOpen ? "set-nav" : "set-nav is-collapsed"}>
            {TABS.map((name) => (
              <button
                key={name}
                className={name === tab ? "set-nav-item active" : "set-nav-item"}
                onClick={() => setTab(name)}
                aria-current={name === tab}
                title={TAB_HINTS[name]}
              >
                <Icon name={TAB_ICONS[name]} size={16} filled={name === tab} />
                <span>{name}</span>
              </button>
            ))}
          </nav>

          <div
            className={scrolled ? "set-content is-scrolled" : "set-content"}
            onScroll={(event) => setScrolled(event.currentTarget.scrollTop > 4)}
          >
            {settingsError && (
              <div className="set-error" role="alert">
                <Icon name="close" size={13} />
                <span>{settingsError}</span>
              </div>
            )}

            {loading ? (
              <div className="modal-loading">
                <span className="mini-spin" />
                Reading settings…
              </div>
            ) : (
              <>
                {tab === "Chat" && (
                  <>
                    <Section title="Checkpoint" hint="The model file this app loads from disk.">
                      <Row label="Model file" hint={`${models.length} found in checkpoints/`}>
                        <Select
                          value={checkpoint}
                          onChange={setCheckpoint}
                          options={[
                            ...(checkpoint === "" ? [{ value: "", label: "— none selected —" }] : []),
                            ...(!models.some((model) => model.path === checkpoint) &&
                            checkpoint !== ""
                              ? [
                                  {
                                    value: checkpoint,
                                    label: checkpoint.split(/[\\/]/).pop() ?? checkpoint,
                                  },
                                ]
                              : []),
                            ...models.map((model) => ({
                              value: model.path,
                              label: `${model.display_name} · ${(model.size / 1e6).toFixed(0)} MB`,
                            })),
                          ]}
                        />
                        <button className="btn small" onClick={() => void browse()}>
                          Browse…
                        </button>
                      </Row>

                      <Row
                        label="Device"
                        hint={
                          device === "cuda" && runningDevice === "cpu"
                            ? "Selected CUDA, but the loaded model reports CPU — this build may lack CUDA support"
                            : cudaAvailable
                              ? "CUDA is available to PyTorch"
                              : "No CUDA device found — the backend runs on CPU"
                        }
                      >
                        <Select
                          value={device}
                          onChange={setDevice}
                          options={[
                            { value: "cpu", label: "CPU" },
                            {
                              value: "cuda",
                              label: cudaAvailable ? "CUDA" : "CUDA (not detected)",
                              disabled: !cudaAvailable,
                            },
                          ]}
                        />
                      </Row>

                      <Row label="GGUF chat format" hint="Template used to wrap the conversation">
                        <Select
                          value={ggufFormat}
                          onChange={setGgufFormat}
                          options={[
                            { value: "auto", label: "auto" },
                            { value: "chatml", label: "chatml" },
                            { value: "llama-2", label: "llama-2" },
                          ]}
                        />
                      </Row>
                    </Section>

                    <Section title="Provider" hint="Where answers come from — local model or a cloud API.">
                      <div className="provider-wrap">
                        <div className="provider-grid">
                          {gridProviders.map((provider) => providerCard(provider))}
                        </div>
                        {customProvider && providerCard(customProvider, true)}
                      </div>

                      {selected && selected.slug === "ollama" && (
                        <div className="key-panel">
                          <div className="key-fields">
                            <select
                              className="set-select grow"
                              value={apiModel}
                              disabled={loadingModels || ollamaModels.length === 0}
                              onChange={(event) => setApiModel(event.target.value)}
                            >
                              {loadingModels && <option value="">Looking for models…</option>}
                              {!loadingModels && ollamaModels.length === 0 && (
                                <option value="">No models found</option>
                              )}
                              {ollamaModels.map((model) => (
                                <option key={model.id} value={model.id}>
                                  {model.name}
                                </option>
                              ))}
                            </select>
                            <button
                              className="btn small"
                              onClick={() => void loadOllamaModels(apiModel)}
                              disabled={loadingModels}
                            >
                              Refresh
                            </button>
                            <button
                              className="btn small primary"
                              onClick={() => void activate("ollama", apiModel)}
                              disabled={!apiModel || activating}
                            >
                              Use Ollama
                            </button>
                          </div>
                          <div className="key-fields">
                            <input
                              className="set-input grow"
                              value={ollamaUrl}
                              onChange={(event) => setOllamaUrl(event.target.value)}
                              placeholder="http://localhost:11434"
                              spellCheck={false}
                              aria-label="Ollama server URL"
                            />
                            <button
                              className="btn small"
                              onClick={() => void applyOllamaUrl()}
                              disabled={activating}
                            >
                              Apply URL
                            </button>
                          </div>
                          <small className={ollamaError ? "is-error" : undefined}>
                            {ollamaError ??
                              `Talks to ${ollamaUrl.trim() || "http://localhost:11434"} — start it with \`ollama serve\`.`}
                            {keyStatus && <b className="key-status"> {keyStatus}</b>}
                          </small>
                        </div>
                      )}

                      {selected && selected.slug === "custom" && (
                        <div className="key-panel">
                          <div className="key-fields">
                            <input
                              className="set-input"
                              value={customName}
                              onChange={(event) => setCustomName(event.target.value)}
                              placeholder="Provider name (e.g. OpenRouter)"
                              spellCheck={false}
                            />
                            <input
                              className="set-input"
                              value={customUrl}
                              onChange={(event) => setCustomUrl(event.target.value)}
                              placeholder="https://openrouter.ai/api/v1"
                              spellCheck={false}
                            />
                          </div>
                          <div className="key-fields">
                            <input
                              className="set-input"
                              type="password"
                              value={apiKey}
                              onChange={(event) => setApiKey(event.target.value)}
                              placeholder="API key"
                              spellCheck={false}
                            />
                            {customModels.length > 0 ? (
                              <select
                                className="set-select grow"
                                value={apiModel}
                                onChange={(event) => setApiModel(event.target.value)}
                              >
                                {customModels.map((model) => (
                                  <option key={model.id} value={model.id}>
                                    {model.id}
                                  </option>
                                ))}
                              </select>
                            ) : (
                              <input
                                className="set-input"
                                value={apiModel}
                                onChange={(event) => setApiModel(event.target.value)}
                                placeholder="model id (optional)"
                                spellCheck={false}
                              />
                            )}
                            <button
                              className="btn small primary"
                              onClick={() => void connectCustom()}
                              disabled={!customUrl.trim() || activating}
                            >
                              {activating ? "Connecting…" : "Save & connect"}
                            </button>
                          </div>
                          <small className={customError ? "is-error" : undefined}>
                            {customError ??
                              "Any OpenAI-compatible endpoint. The base URL usually ends in /v1; models are pulled automatically."}
                            {keyStatus && <b className="key-status"> {keyStatus}</b>}
                          </small>
                        </div>
                      )}

                      {selected && selected.requires_key && selected.slug !== "custom" && (
                        <div className="key-panel">
                          <div className="key-fields">
                            <input
                              className="set-input"
                              type="password"
                              value={apiKey}
                              onChange={(event) => setApiKey(event.target.value)}
                              placeholder={`${selected.name} API key`}
                              spellCheck={false}
                            />
                            <input
                              className="set-input"
                              value={apiModel}
                              onChange={(event) => setApiModel(event.target.value)}
                              placeholder="model id (optional)"
                              spellCheck={false}
                            />
                            <button
                              className="btn small primary"
                              onClick={() => void activate(activeProvider, apiModel, apiKey)}
                              disabled={activating}
                            >
                              Save key
                            </button>
                          </div>
                          <small>
                            Stored encrypted in <code>.dizel/settings.json</code> and applied
                            immediately.
                            {keyStatus && <b className="key-status"> {keyStatus}</b>}
                          </small>
                        </div>
                      )}
                    </Section>

                    <Section title="Context window">
                      <Row
                        label="Provider context length"
                        hint="Used when a provider model answers — they don't report their own"
                      >
                        <input
                          className="set-number"
                          type="number"
                          min={512}
                          max={1000000}
                          step={512}
                          value={ctxLength}
                          onChange={(event) => setCtxLength(event.target.value)}
                        />
                      </Row>
                    </Section>

                    <Section
                      title="Memory & length"
                      hint="How much conversation the model keeps, and how long replies may run."
                    >
                      {rangeRow(
                        "max-context",
                        "Context memory",
                        "Tokens of history kept before older turns are dropped",
                        budget.max_context_tokens,
                        2000,
                        32000,
                        500,
                        (v) => `${(v / 1000).toFixed(1)}k`,
                        (v) => setBudget((prev) => ({ ...prev, max_context_tokens: v })),
                      )}
                      <Row
                        label="Verbosity"
                        hint="Scales the reply budget for every task type"
                      >
                        <Select
                          value={budget.verbosity}
                          onChange={(value) =>
                            setBudget((prev) => ({ ...prev, verbosity: value }))
                          }
                          options={[
                            { value: "low", label: "Concise" },
                            { value: "normal", label: "Balanced" },
                            { value: "high", label: "Detailed" },
                          ]}
                        />
                      </Row>
                      <Row
                        label="Hard reply cap"
                        hint="Absolute token ceiling per reply · 0 lets the task budget decide"
                      >
                        <input
                          className="set-number"
                          type="number"
                          min={0}
                          max={8192}
                          step={32}
                          value={hardCap}
                          aria-label="Hard reply cap in tokens"
                          onChange={(event) => setHardCap(event.target.value)}
                          onBlur={() =>
                            setHardCap(String(Math.min(8192, Math.max(0, Number(hardCap) || 0))))
                          }
                        />
                      </Row>
                    </Section>

                    <Section title="Tools" hint="What the model may reach for while answering.">
                      <Row
                        label="Let the model call tools"
                        hint={
                          tools?.enabled
                            ? "Offered on every request — costs prompt tokens"
                            : "Off by default; enabling adds ~330 tokens per request"
                        }
                      >
                        <Switch
                          checked={Boolean(tools?.enabled)}
                          label="Let the model call tools"
                          onChange={() => {
                            void api
                              .setTools(!tools?.enabled)
                              .then(setTools)
                              .catch(() => undefined);
                          }}
                        />
                      </Row>
                      <div className="field-block">
                        <label>Available</label>
                        {/* Inline code per tool, the way `name` reads in prose —
                            not a fenced block. */}
                        <div className="set-inline-list">
                          {(tools?.available ?? []).map((name) => (
                            <code key={name} className="set-inline-code">
                              {name}
                            </code>
                          ))}
                        </div>
                      </div>
                    </Section>

                    <Section title="Usage window" hint="A 12-hour budget that resets on its own.">
                      <Row
                        label="Spent this window"
                        hint={
                          usage
                            ? `${usage.percent}% of ${usage.capacity.toLocaleString()} · resets in ${usage.resets_in}`
                            : "Reading…"
                        }
                      >
                        <span className={usage?.exhausted ? "set-value is-warn" : "set-value"}>
                          {usage
                            ? `${usage.used.toLocaleString()} / ${usage.capacity.toLocaleString()}`
                            : "—"}
                        </span>
                      </Row>
                      <Row
                        label="Compactions"
                        hint="Automatic once the window passes 84%"
                      >
                        <span className="set-value">
                          {usage
                            ? `${usage.compactions_used} used · ${usage.compactions_remaining} left`
                            : "—"}
                        </span>
                      </Row>
                    </Section>

                    <Section title="System prompt">
                      {Object.keys(personas).length > 0 && (
                        <div className="field-block">
                          <label>Persona presets</label>
                          <div className="persona-row">
                            {Object.entries(personas).map(([name, text]) => (
                              <button
                                key={name}
                                className="persona-chip"
                                aria-pressed={systemPrompt === text}
                                onClick={() => setSystemPrompt(text)}
                                title={text}
                              >
                                {name}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                      <div className="field-block">
                        <label htmlFor="system-prompt">
                          Instructions prepended to every message
                        </label>
                        <textarea
                          id="system-prompt"
                          className="set-textarea"
                          value={systemPrompt}
                          onChange={(event) => setSystemPrompt(event.target.value)}
                          rows={4}
                          placeholder="You are Dizel, a careful assistant…"
                          spellCheck={false}
                        />
                      </div>
                    </Section>

                    <Section
                      title="Sampling"
                      action={
                        <button className="link-btn" onClick={() => setSampling(DEFAULT_SAMPLING)}>
                          Reset
                        </button>
                      }
                    >
                      {slider("temperature", "Temperature", 0, 2, 0.05, (v) => v.toFixed(2))}
                      {slider("top_k", "Top K", 0, 100, 1, (v) => String(v))}
                      {slider("top_p", "Top P", 0.1, 1, 0.01, (v) => v.toFixed(2))}
                      {slider("repetition_penalty", "Repetition penalty", 1, 2, 0.01, (v) =>
                        v.toFixed(2),
                      )}
                      {slider("max_new_tokens", "Max new tokens", 32, 2048, 32, (v) => String(v))}
                    </Section>
                  </>
                )}

                {tab === "MCP" && <McpSettings />}

                {tab === "Model" && (
                  <>
                    <Section
                      title="Response budgets"
                      hint="Base tokens per task type — the classifier picks one for each prompt, then verbosity scales it."
                    >
                      {rangeRow(
                        "chat-budget",
                        "Chat",
                        "Everyday conversation and quick questions",
                        budget.chat_budget,
                        32,
                        1024,
                        16,
                        (v) => String(v),
                        (v) => setBudget((prev) => ({ ...prev, chat_budget: v })),
                      )}
                      {rangeRow(
                        "coding-budget",
                        "Coding",
                        "Writing, fixing and explaining code",
                        budget.coding_budget,
                        64,
                        2048,
                        32,
                        (v) => String(v),
                        (v) => setBudget((prev) => ({ ...prev, coding_budget: v })),
                      )}
                      {rangeRow(
                        "complex-budget",
                        "Complex reasoning",
                        "Multi-step analysis, math and planning",
                        budget.complex_budget,
                        64,
                        2048,
                        32,
                        (v) => String(v),
                        (v) => setBudget((prev) => ({ ...prev, complex_budget: v })),
                      )}
                      {rangeRow(
                        "factual-budget",
                        "Factual",
                        "Definitions, dates and short lookups",
                        budget.factual_budget,
                        32,
                        1024,
                        16,
                        (v) => String(v),
                        (v) => setBudget((prev) => ({ ...prev, factual_budget: v })),
                      )}
                      {rangeRow(
                        "tool-budget",
                        "Tool use",
                        "Prompts that attach files or call tools",
                        budget.tool_budget,
                        64,
                        2048,
                        32,
                        (v) => String(v),
                        (v) => setBudget((prev) => ({ ...prev, tool_budget: v })),
                      )}
                    </Section>

                    <Section title="Architecture">
                      <Row spec label="Family" onClick={handleFamilyClick}>
                        <b>{facts?.architecture ?? "—"}</b>
                      </Row>
                      <Row spec label="Parameters">
                        <b>{facts?.model_size ?? "—"}</b>
                      </Row>
                      <Row spec label="Context length">
                        <b>{facts?.context_length ?? "—"}</b>
                      </Row>
                      <Row spec label="Vocab size">
                        <b>{facts?.vocab_size ?? "—"}</b>
                      </Row>
                      <Row spec label="Layers / heads / dim">
                        <b>
                          {facts
                            ? `${facts.num_layers} / ${facts.num_heads} (kv ${facts.kv_heads}) / ${facts.hidden_dim}`
                            : "—"}
                        </b>
                      </Row>
                    </Section>

                    <Section title="Training data">
                      <Row spec label="Dataset composition">
                        <b>{facts?.training_data ?? "—"}</b>
                      </Row>
                      <Row spec label="Corpus size">
                        <b>{facts?.train_tokens ?? "—"}</b>
                      </Row>
                      <Row spec label="Training steps">
                        <b>{facts?.train_steps ?? "—"}</b>
                      </Row>
                    </Section>

                    <Section title="Capabilities">
                      <div className="cap-list">
                        {(facts?.capabilities ?? []).map((capability) => (
                          <div className="cap-row" key={capability}>
                            <span className="cap-dot" />
                            {capability}
                          </div>
                        ))}
                        {!facts?.capabilities?.length && <small>No data.</small>}
                      </div>
                    </Section>
                  </>
                )}

                {tab === "Appearance" && (
                  <>
                    <Section title="Theme" hint="Colours, accent and type for the whole app.">
                      <Row label="Color mode">
                        <Segmented
                          ariaLabel="Color mode"
                          value={theme}
                          onChange={(value) => setTheme(value as ThemeName)}
                          options={THEMES}
                        />
                      </Row>
                      <Row label="Font scale" hint="Applies on save">
                        <Select
                          value={fontScale}
                          onChange={(value) => setFontScale(value as FontScale)}
                          options={FONT_SCALES}
                        />
                      </Row>

                      <Row label="Accent" hint="Buttons, links and the active model">
                        <div className="swatch-row">
                          {(["blue", "teal", "violet", "rose"] as const).map((name) => (
                            <button
                              key={name}
                              className={accent === name ? "swatch active" : "swatch"}
                              data-swatch={name}
                              onClick={() => setAccent(name)}
                              title={name}
                              aria-label={name}
                            />
                          ))}
                        </div>
                      </Row>

                      <Row label="Density" hint="How much room each message takes">
                        <Select
                          value={density}
                          onChange={(value) => setDensity(value as Density)}
                          options={[
                            { value: "comfortable", label: "Comfortable" },
                            { value: "compact", label: "Compact" },
                          ]}
                        />
                      </Row>

                      <Row label="Message font" hint="Typeface for replies">
                        <Select
                          value={fontFamily}
                          onChange={(value) => setFontFamily(value as FontFamily)}
                          options={[
                            { value: "system", label: "System" },
                            { value: "serif", label: "Serif" },
                            { value: "mono", label: "Mono" },
                          ]}
                        />
                      </Row>
                    </Section>

                    <Section title="Messages" hint="How replies are rendered in the transcript.">
                      <ToggleRow
                        label="Show timestamps"
                        hint="Time beside each reply's token count"
                        checked={showTimestamps}
                        onChange={setShowTimestamps}
                      />
                      <ToggleRow
                        label="24-hour clock"
                        hint={clock24h ? "14:05" : "2:05 PM"}
                        checked={clock24h}
                        onChange={setClock24h}
                      />
                      <ToggleRow
                        label="Show reply speed"
                        hint="Estimated tokens per second beside each reply"
                        checked={showSpeed}
                        onChange={setShowSpeed}
                      />
                      <ToggleRow
                        label="Show model tag"
                        hint="Which model answered, in each reply's meta line"
                        checked={showModelTag}
                        onChange={setShowModelTag}
                      />
                      <Row
                        label="Typing animation"
                        hint="How quickly streamed text catches up to the model"
                      >
                        <Select
                          value={typingAnimation}
                          onChange={(value) =>
                            setTypingAnimation(
                              value as "off" | "fast" | "smooth" | "cinematic",
                            )
                          }
                          options={[
                            { value: "off", label: "Off" },
                            { value: "fast", label: "Fast" },
                            { value: "smooth", label: "Smooth" },
                            { value: "cinematic", label: "Cinematic" },
                          ]}
                        />
                      </Row>
                      <Row label="Line height" hint="Breathing room between lines of text">
                        <Select
                          value={lineHeight}
                          onChange={(value) => setLineHeight(value as "normal" | "relaxed")}
                          options={[
                            { value: "normal", label: "Normal" },
                            { value: "relaxed", label: "Relaxed" },
                          ]}
                        />
                      </Row>
                      <ToggleRow
                        label="Underline links"
                        hint="Links in replies keep an underline"
                        checked={underlineLinks}
                        onChange={setUnderlineLinks}
                      />
                      <ToggleRow
                        label="Show avatars"
                        hint="Model mark beside each reply"
                        checked={avatars}
                        onChange={setAvatars}
                      />
                      <Row label="Thinking pause" hint="Hold the answer behind the dots">
                        <Select
                          value={thinkingPause}
                          onChange={(value) =>
                            setThinkingPause(value as "off" | "short" | "long" | "custom")
                          }
                          options={[
                            { value: "off", label: "Off" },
                            { value: "short", label: "Short (scales with the prompt)" },
                            { value: "long", label: "Long (10s)" },
                            { value: "custom", label: "Custom…" },
                          ]}
                        />
                      </Row>
                      {thinkingPause === "custom" && (
                        <Row
                          label="Custom pause"
                          hint="Seconds to hold the answer · 1 hour = 3600"
                        >
                          <input
                            className="set-number"
                            type="number"
                            min={1}
                            max={86400}
                            step={1}
                            value={customPause}
                            aria-label="Custom thinking pause in seconds"
                            onChange={(event) => setCustomPause(event.target.value)}
                            onBlur={() =>
                              setCustomPause(String(customPauseSeconds(customPause)))
                            }
                          />
                        </Row>
                      )}
                      <ToggleRow
                        label="Syntax highlighting"
                        hint="Colour code blocks"
                        checked={syntaxHighlight}
                        onChange={setSyntaxHighlight}
                      />
                      <ToggleRow
                        label="UI animations"
                        hint="Message fade-in, shimmer, hover motion"
                        checked={animations}
                        onChange={setAnimations}
                      />
                      <ToggleRow
                        label="Press Enter to send"
                        hint={enterToSend ? "Shift+Enter adds a line" : "Ctrl+Enter sends instead"}
                        checked={enterToSend}
                        onChange={setEnterToSend}
                      />
                    </Section>

                    <Section title="Layout" hint="How the transcript fills the window.">
                      <Row label="Message width" hint="How wide the conversation runs">
                        <Select
                          value={messageWidth}
                          onChange={(value) =>
                            setMessageWidth(value as "comfortable" | "wide" | "full")
                          }
                          options={[
                            { value: "comfortable", label: "Comfortable" },
                            { value: "wide", label: "Wide" },
                            { value: "full", label: "Full width" },
                          ]}
                        />
                      </Row>
                      <ToggleRow
                        label="Separators"
                        hint="A hairline between messages"
                        checked={separators}
                        onChange={setSeparators}
                      />
                      <ToggleRow
                        label="Wrap code"
                        hint="Long lines wrap instead of scrolling sideways"
                        checked={codeWrap}
                        onChange={setCodeWrap}
                      />
                      <Row label="Message spacing" hint="Gap between turns">
                        <Select
                          value={messageSpacing}
                          onChange={(value) =>
                            setMessageSpacing(value as "comfortable" | "compact")
                          }
                          options={[
                            { value: "comfortable", label: "Comfortable" },
                            { value: "compact", label: "Compact" },
                          ]}
                        />
                      </Row>
                      <Row label="Message corners" hint="Shape of your message bubbles">
                        <Select
                          value={messageCorners}
                          onChange={(value) =>
                            setMessageCorners(value as "rounded" | "soft" | "sharp")
                          }
                          options={[
                            { value: "rounded", label: "Rounded" },
                            { value: "soft", label: "Soft" },
                            { value: "sharp", label: "Sharp" },
                          ]}
                        />
                      </Row>
                    </Section>

                    <Section title="Behaviour" hint="Small things that change how the app feels.">
                      <ToggleRow
                        label="Follow new messages"
                        hint="Keep the transcript pinned to the newest reply while it streams"
                        checked={autoScroll}
                        onChange={setAutoScroll}
                      />
                      <ToggleRow
                        label="Spellcheck"
                        hint="Browser spellcheck while typing"
                        checked={spellcheck}
                        onChange={setSpellcheck}
                      />
                      <ToggleRow
                        label="Starter suggestions"
                        hint="Show prompt pills on an empty chat"
                        checked={showSuggestions}
                        onChange={setShowSuggestions}
                      />
                      <ToggleRow
                        label="Flash taskbar on reply"
                        hint="When a reply lands while the window is in the background"
                        checked={notifyOnFinish}
                        onChange={setNotifyOnFinish}
                      />
                      <ToggleRow
                        label="Reduce transparency"
                        hint="Solid panels instead of blurred glass — easier on the eyes and GPU"
                        checked={reduceTransparency}
                        onChange={setReduceTransparency}
                      />
                    </Section>
                  </>
                )}

                {tab === "Speech" && (
                  <>
                    <Section title="Nova dictation" hint="Speech-to-text, running on this machine.">
                      <Row
                        label="Whisper model"
                        hint="Larger is more accurate but loads slower and needs more RAM"
                      >
                        <Select
                          value={whisperSize}
                          onChange={setWhisperSize}
                          options={[
                            { value: "tiny", label: "tiny · 39M" },
                            { value: "base", label: "base · 74M" },
                            { value: "small", label: "small · 244M" },
                            { value: "medium", label: "medium · 769M" },
                            { value: "large-v3", label: "large-v3 · 1.55G" },
                          ]}
                        />
                      </Row>
                      <Row label="Language" hint="Auto lets Whisper detect it">
                        <Select
                          value={language}
                          onChange={setLanguage}
                          options={[
                            { value: "auto", label: "auto" },
                            { value: "en", label: "English" },
                            { value: "vi", label: "Vietnamese" },
                            { value: "fr", label: "French" },
                            { value: "de", label: "German" },
                            { value: "es", label: "Spanish" },
                            { value: "it", label: "Italian" },
                            { value: "pt", label: "Portuguese" },
                            { value: "nl", label: "Dutch" },
                            { value: "ru", label: "Russian" },
                            { value: "uk", label: "Ukrainian" },
                            { value: "pl", label: "Polish" },
                            { value: "tr", label: "Turkish" },
                            { value: "ar", label: "Arabic" },
                            { value: "hi", label: "Hindi" },
                            { value: "zh", label: "Chinese" },
                            { value: "ja", label: "Japanese" },
                            { value: "ko", label: "Korean" },
                            { value: "th", label: "Thai" },
                            { value: "id", label: "Indonesian" },
                            { value: "sv", label: "Swedish" },
                            { value: "da", label: "Danish" },
                            { value: "no", label: "Norwegian" },
                            { value: "fi", label: "Finnish" },
                            { value: "el", label: "Greek" },
                            { value: "he", label: "Hebrew" },
                            { value: "cs", label: "Czech" },
                            { value: "ro", label: "Romanian" },
                            { value: "hu", label: "Hungarian" },
                          ]}
                        />
                      </Row>
                      <Row
                        label="Task"
                        hint="Transcribe keeps the source language; Translate returns English"
                      >
                        <Select
                          value={whisperTask}
                          onChange={setWhisperTask}
                          options={[
                            { value: "transcribe", label: "Transcribe" },
                            { value: "translate", label: "Translate to English" },
                          ]}
                        />
                      </Row>
                    </Section>

                    <Section
                      title="Decoding"
                      hint="How Whisper searches for the transcript. Defaults match the model; tune for speed or accuracy."
                    >
                      {rangeRow(
                        "beam-size",
                        "Beam size",
                        "Hypotheses kept in parallel while decoding",
                        beamSize,
                        1,
                        10,
                        1,
                        (v) => String(v),
                        setBeamSize,
                      )}
                      {rangeRow(
                        "best-of",
                        "Best of",
                        "Candidates sampled when temperature is above 0",
                        bestOf,
                        1,
                        10,
                        1,
                        (v) => String(Math.max(beamSize, v)),
                        setBestOf,
                      )}
                      {rangeRow(
                        "whisper-temp",
                        "Temperature",
                        "0 is deterministic; higher adds variation",
                        whisperTemp,
                        0,
                        1,
                        0.05,
                        (v) => v.toFixed(2),
                        setWhisperTemp,
                      )}
                      {rangeRow(
                        "no-speech",
                        "No-speech threshold",
                        "Confidence required to keep a segment as speech",
                        noSpeechThreshold,
                        0,
                        1,
                        0.05,
                        (v) => v.toFixed(2),
                        setNoSpeechThreshold,
                      )}
                      <ToggleRow
                        label="Condition on previous text"
                        hint="Carry the earlier transcript forward as context"
                        checked={conditionPrev}
                        onChange={setConditionPrev}
                      />
                    </Section>

                    <Section
                      title="Vocabulary"
                      hint="Names, acronyms and jargon to bias the next dictation."
                    >
                      <div className="field-block">
                        <label htmlFor="whisper-prompt">Initial prompt</label>
                        <textarea
                          id="whisper-prompt"
                          className="set-textarea"
                          rows={2}
                          maxLength={400}
                          value={whisperPrompt}
                          onChange={(event) => setWhisperPrompt(event.target.value)}
                          placeholder="Dizel, GGUF, safetensors, Dan…"
                          spellCheck={false}
                        />
                      </div>
                    </Section>
                  </>
                )}

                {tab === "Shortcuts" && (
                  <>
                    <Section
                      title="Keyboard behavior"
                      hint="Optional bindings and composer habits — toggle what suits you."
                    >
                      <ToggleRow
                        label="Auto-focus the composer"
                        hint="Jump to the input when a chat opens or the app is ready"
                        checked={autoFocusComposer}
                        onChange={setAutoFocusComposer}
                      />
                      <ToggleRow
                        label="Escape clears the draft"
                        hint="Esc wipes the composer while it's focused"
                        checked={escapeClearsDraft}
                        onChange={setEscapeClearsDraft}
                      />
                      <ToggleRow
                        label="Arrow Up edits your last message"
                        hint="↑ in an empty composer reopens your previous prompt"
                        checked={arrowUpEditsLast}
                        onChange={setArrowUpEditsLast}
                      />
                      <ToggleRow
                        label="Copy the last reply"
                        hint="Ctrl+Shift+C copies the newest answer"
                        checked={copyLastReplyHotkey}
                        onChange={setCopyLastReplyHotkey}
                      />
                      <ToggleRow
                        label="Stop generation"
                        hint="Ctrl+Shift+X halts the current reply"
                        checked={stopGenerationHotkey}
                        onChange={setStopGenerationHotkey}
                      />
                    </Section>

                    <Section title="Global">
                      <Row label="New chat">
                        <span className="keys">
                          <kbd>Ctrl</kbd>
                          <kbd>N</kbd>
                        </span>
                      </Row>
                      <Row label="Focus the composer">
                        <span className="keys">
                          <kbd>Ctrl</kbd>
                          <kbd>K</kbd>
                        </span>
                      </Row>
                      <Row label="Toggle the sidebar">
                        <span className="keys">
                          <kbd>Ctrl</kbd>
                          <kbd>B</kbd>
                        </span>
                      </Row>
                      <Row label="Open settings">
                        <span className="keys">
                          <kbd>Ctrl</kbd>
                          <kbd>,</kbd>
                        </span>
                      </Row>
                      <Row label="Close a dialog">
                        <span className="keys">
                          <kbd>Esc</kbd>
                        </span>
                      </Row>
                      <Row label="Toggle fullscreen">
                        <span className="keys">
                          <kbd>F11</kbd>
                        </span>
                      </Row>
                    </Section>

                    <Section title="Composer">
                      <Row label="Send message">
                        <span className="keys">
                          {enterToSend ? (
                            <kbd>Enter</kbd>
                          ) : (
                            <>
                              <kbd>Ctrl</kbd>
                              <kbd>Enter</kbd>
                            </>
                          )}
                        </span>
                      </Row>
                      <Row label="New line">
                        <span className="keys">
                          {enterToSend ? (
                            <>
                              <kbd>Shift</kbd>
                              <kbd>Enter</kbd>
                            </>
                          ) : (
                            <kbd>Enter</kbd>
                          )}
                        </span>
                      </Row>
                      <Row label="Rename a chat">
                        <span className="keys">
                          <kbd>Double-click</kbd>
                        </span>
                      </Row>
                    </Section>
                  </>
                )}

                {tab === "About" && (
                  <>
                    <div className="about-head">
                      <Logo size={64} />
                      <h2>{facts?.app_name ?? "Dizel"}</h2>
                      <p>
                        {appVersion ? `App ${appVersion}` : "App —"}
                        {facts?.version ? ` · Model ${facts.version}` : ""}
                        {facts?.build ? ` · Build ${facts.build}` : ""}
                      </p>
                    </div>

                    <Section
                      title="Updates"
                      hint="Stable builds published on GitHub Releases."
                    >
                      {update ? (
                        <>
                          <Row label="Installed">
                            <b>{update.currentVersion}</b>
                          </Row>
                          <Row
                            label="Check automatically"
                            hint="Every six hours while Dizel is running."
                          >
                            <Switch
                              checked={update.autoCheck}
                              onChange={() =>
                                void window.dizel?.updates?.setAutoCheck(!update.autoCheck)
                              }
                              label="Check for updates automatically"
                            />
                          </Row>
                          <div
                            className={`update-card is-${update.status}`}
                            role="status"
                            aria-live="polite"
                          >
                            <div className="update-card-head">
                              <span className="update-mark" aria-hidden="true">
                                <Icon name={updateMark} size={16} />
                              </span>
                              <div className="update-heading">
                                <span className="update-version">{updateTitle}</span>
                                <span className="update-channel">
                                  GitHub Releases · Stable channel
                                </span>
                              </div>
                            </div>
                            <p className="update-text">{updateText}</p>
                            {update.status === "downloading" && (
                              <div
                                className="update-meter"
                                role="progressbar"
                                aria-valuemin={0}
                                aria-valuemax={100}
                                aria-valuenow={update.progress}
                                aria-label="Update download progress"
                              >
                                <span style={{ width: `${update.progress}%` }} />
                              </div>
                            )}
                            {(update.status === "available" ||
                              update.status === "downloading" ||
                              update.status === "downloaded") &&
                              update.notes && (
                                <p className="update-notes">
                                  {update.notes.slice(0, 600)}
                                  {update.notes.length > 600 ? "…" : ""}
                                </p>
                              )}
                            <div className="update-actions">
                              <button
                                className="btn small ghost"
                                onClick={() => void window.dizel?.updates?.check()}
                                disabled={updateBusy}
                              >
                                {update.status === "checking" ? "Checking…" : "Check now"}
                              </button>
                              {update.status === "available" && (
                                <button
                                  className="btn small primary"
                                  onClick={() => void window.dizel?.updates?.download()}
                                >
                                  {update.dev ? "Open release page" : "Download update"}
                                </button>
                              )}
                              {update.status === "downloaded" && (
                                <button
                                  className="btn small primary"
                                  onClick={() => void window.dizel?.updates?.install()}
                                >
                                  Restart &amp; install
                                </button>
                              )}
                              {update.latestVersion && (
                                <button
                                  className="btn small ghost"
                                  onClick={() => void window.dizel?.updates?.openRelease()}
                                >
                                  Release notes
                                </button>
                              )}
                            </div>
                            {update.dev && (
                              <p className="update-dev">
                                Development build — installing opens the release page.
                              </p>
                            )}
                          </div>
                        </>
                      ) : (
                        <Row label="Updates">
                          <b>Desktop app only</b>
                        </Row>
                      )}
                    </Section>

                    <Section title="Links">
                      <button
                        className="set-row as-button"
                        onClick={() => void window.dizel?.openExternal(facts?.repository ?? "")}
                      >
                        <span className="set-row-text">
                          <span>GitHub repository</span>
                          <small>{facts?.repository ?? "—"}</small>
                        </span>
                        <Icon name="code" size={14} />
                      </button>
                      <Row label="License">
                        <b>{facts?.license ?? "—"}</b>
                      </Row>
                    </Section>

                    <Section title="Runtime" hint="How the app starts and behaves.">
                      <Row label="Checkpoint">
                        <b>{checkpoint.split(/[\\/]/).pop() || "—"}</b>
                      </Row>
                      <Row label="Device">
                        <b>{device.toUpperCase()}</b>
                      </Row>
                      <Row label="Saved chats">
                        <b>{sessionCount}</b>
                      </Row>
                      <Row label="Build">
                        <b>{facts?.build ?? "—"}</b>
                      </Row>
                    </Section>

                    <Section
                      title="Backup & diagnostics"
                      hint="Move settings between machines or capture what this build is doing."
                    >
                      <button
                        className="set-row as-button"
                        onClick={() => void exportSettings()}
                      >
                        <span className="set-row-text">
                          <span>Export settings</span>
                          <small>Saves a JSON copy — MCP secrets are left out</small>
                        </span>
                        <Icon name="download" size={14} />
                      </button>
                      <button
                        className="set-row as-button"
                        onClick={() => void importSettings()}
                      >
                        <span className="set-row-text">
                          <span>Import settings</span>
                          <small>
                            Restores a JSON export and applies it right away; MCP servers stay put
                          </small>
                        </span>
                        <Icon name="folder" size={14} />
                      </button>
                      <button
                        className="set-row as-button"
                        onClick={() => void copyDiagnostics()}
                      >
                        <span className="set-row-text">
                          <span>Copy diagnostics</span>
                          <small>Version, model, device and preferences to the clipboard</small>
                        </span>
                        <Icon name="copy" size={14} />
                      </button>
                      <div className="set-row">
                        <div className="set-row-text">
                          <span>Clear UI cache</span>
                          <small>
                            Resets window state like theme and layout, then reloads. Chats and
                            settings stay.
                          </small>
                        </div>
                        <div className="set-row-control">
                          {confirmCache ? (
                            <>
                              <button className="btn small" onClick={() => setConfirmCache(false)}>
                                Cancel
                              </button>
                              <button className="btn small danger" onClick={clearUiCache}>
                                Clear & reload
                              </button>
                            </>
                          ) : (
                            <button className="btn small" onClick={() => setConfirmCache(true)}>
                              Clear cache
                            </button>
                          )}
                        </div>
                      </div>
                      <div className="set-row">
                        <div className="set-row-text">
                          <span>Reset preferences</span>
                          <small>
                            Restores appearance, sampling, budgets and dictation. Keeps your
                            model, provider and chats.
                          </small>
                        </div>
                        <div className="set-row-control">
                          {confirmReset ? (
                            <>
                              <button className="btn small" onClick={() => setConfirmReset(false)}>
                                Cancel
                              </button>
                              <button
                                className="btn small danger"
                                onClick={() => void resetPreferences()}
                              >
                                Reset all
                              </button>
                            </>
                          ) : (
                            <button className="btn small danger" onClick={() => setConfirmReset(true)}>
                              Reset preferences
                            </button>
                          )}
                        </div>
                      </div>
                    </Section>

                    <Section title="Data & privacy" hint="What is stored, and where.">
                      <div className="set-row">
                        <div className="set-row-text">
                          <span>Delete all saved chats</span>
                          <small>
                            Chats live as JSON in Dizel&apos;s local history folder. Cannot be
                            undone.
                          </small>
                        </div>
                        <div className="set-row-control">
                          {confirmClear ? (
                            <>
                              <button className="btn small" onClick={() => setConfirmClear(false)}>
                                Cancel
                              </button>
                              <button
                                className="btn small danger"
                                onClick={() => void clearChats()}
                              >
                                Delete all
                              </button>
                            </>
                          ) : (
                            <button
                              className="btn small danger"
                              onClick={() => setConfirmClear(true)}
                            >
                              Clear chats
                            </button>
                          )}
                        </div>
                      </div>
                      {notice && <div className="set-note">{notice}</div>}
                    </Section>
                  </>
                )}
              </>
            )}
          </div>
        </div>

        <footer className="set-foot">
          <button
            className="btn ghost"
            onClick={resetSection}
            disabled={
              loading ||
              tab === "MCP" ||
              tab === "About"
            }
          >
            Reset {tab.toLowerCase()}
          </button>
          <span className="spacer" />
          <button className="btn" onClick={close}>
            Cancel
          </button>
          <button className="btn primary" onClick={submit} disabled={loading}>
            {loading ? "Saving…" : "Save & close"}
          </button>
        </footer>
      </div>
    </div>
  );
}
