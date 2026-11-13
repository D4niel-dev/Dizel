/* Dizel — browser demo of the desktop app.
   The shell mirrors inference/gui/src; the model is replaced by a scripted
   responder so the page needs no Python, no network and no keys.

   Composition, in the same order as inference/gui/src/App.tsx:
     titlebar → sidebar → secondary sidebar → main (header, view, composer, status)
   The composer is a single node that moves between the welcome hero and the
   docked position, exactly like ChatView.tsx / App.tsx do. */

(function () {
  "use strict";

  const KEY = "dizel-demo-v5";
  const TIERS = ["Flash", "Pro"];
  const MODES = ["Fast", "Planning", "Coding", "Thinking", "Deep Think"];
  const THEMES = [
    ["dark", "Dark"],
    ["light", "Light"],
    ["dark-blue", "Dark blue"],
  ];
  const SPEECH_MODELS = [
    ["tiny", "tiny · 39M"],
    ["base", "base · 74M"],
    ["small", "small · 244M"],
    ["medium", "medium · 769M"],
    ["large-v3", "large-v3 · 1.55G"],
  ];
  const SPEECH_LANGUAGES = [
    ["auto", "Auto"],
    ["en", "English"],
    ["vi", "Vietnamese"],
    ["fr", "French"],
    ["de", "German"],
    ["es", "Spanish"],
    ["it", "Italian"],
    ["pt", "Portuguese"],
    ["nl", "Dutch"],
    ["ru", "Russian"],
    ["uk", "Ukrainian"],
    ["pl", "Polish"],
    ["tr", "Turkish"],
    ["ar", "Arabic"],
    ["hi", "Hindi"],
    ["zh", "Chinese"],
    ["ja", "Japanese"],
    ["ko", "Korean"],
    ["th", "Thai"],
    ["id", "Indonesian"],
    ["sv", "Swedish"],
    ["da", "Danish"],
    ["no", "Norwegian"],
    ["fi", "Finnish"],
    ["el", "Greek"],
    ["he", "Hebrew"],
    ["cs", "Czech"],
    ["ro", "Romanian"],
    ["hu", "Hungarian"],
  ];
  const SPEECH_TASKS = [
    ["transcribe", "Transcribe"],
    ["translate", "Translate to English"],
  ];
  const SPEECH_DEFAULTS = {
    model: "base",
    language: "auto",
    task: "transcribe",
    prompt: "",
  };
  const RECENT_INITIAL = 20;
  const RECENT_PAGE = 20;
  const RECENT_MAX = 40;
  const WORKSPACES = [
    ["Image", "image", "Image"],
    ["Presentation", "layout", "Presentation"],
    ["Riset", "search", "Riset"],
    ["Archived", "archive", "Archived"],
    ["Library", "book", "Library"],
    ["Monitor", "activity", "Monitor"],
  ];
  const VIEW_TITLES = {
    Chats: "Chats",
    Image: "Images",
    Presentation: "Presentations",
    Riset: "Riset",
    Archived: "Archived",
    Library: "Library",
    Monitor: "Monitor",
  };
  const CONTEXT_LEN = 2048;
  const MODEL = {
    name: "Dizel 1.5 Distill",
    params: "454.78 M",
    dims: "d=1024 L=28",
    device: "DEMO",
  };
  const SUGGESTIONS = [
    { label: "Write Code", icon: "code", prompt: "Write a function that " },
    { label: "Explain a concept", icon: "book", prompt: "Explain how " },
    { label: "Make a plan", icon: "sparkles", prompt: "Make a plan for " },
    { label: "Summarise", icon: "archive", prompt: "Summarise this: " },
    { label: "Translate", icon: "globe", prompt: "Translate to English: " },
    { label: "Fix an error", icon: "alert", prompt: "Why does this fail: " },
  ];
  // Same pools as inference/gui/src/services/welcome.ts. Every headline is a
  // greeting; {greeting} and {name} are filled at render time.
  const HEADLINES = {
    morning: [
      "{greeting}, {name}!",
      "Morning, {name}.",
      "Morning, {name}!",
      "Early start, {name}?",
      "Early bird, {name}?",
      "Early riser, {name}?",
      "Good morning to you, {name}.",
      "A fresh morning, {name}.",
      "New day, {name}.",
      "Morning already, {name}?",
      "Good morning, {name}.",
      "Top of the morning, {name}.",
      "Mornin', {name}.",
      "Morning to you, {name}.",
      "Bright and early, {name}.",
      "A new day, {name}.",
    ],
    afternoon: [
      "{greeting}, {name}!",
      "Afternoon, {name}.",
      "Afternoon, {name}!",
      "Good afternoon to you, {name}.",
      "Good afternoon, {name}.",
      "Afternoon already, {name}?",
      "Back at it, {name}?",
      "Halfway through, {name}?",
      "How's the afternoon, {name}?",
      "Having a good afternoon, {name}?",
      "Good day, {name}.",
      "Good day to you, {name}.",
      "Afternoon to you, {name}.",
      "Mid-afternoon, {name}.",
      "The afternoon's here, {name}.",
      "Still at it, {name}?",
    ],
    evening: [
      "{greeting}, {name}!",
      "Evening, {name}.",
      "Evening, {name}!",
      "Good evening to you, {name}.",
      "Good evening, {name}.",
      "Evening already, {name}?",
      "Still going, {name}?",
      "Still up, {name}?",
      "How's the evening, {name}?",
      "Having a good evening, {name}?",
      "Winding down, {name}?",
      "A good evening to you, {name}.",
      "Evening to you, {name}.",
      "Late evening, {name}.",
      "The evening's here, {name}.",
      "Hope your evening's going well, {name}.",
    ],
  };
  const SUBLINES = [
    "What's on your mind?",
    "Ask anything. Start anywhere.",
    "A question, a task, or a half-formed idea.",
    "Write, debug, plan, or think out loud.",
    "Paste the messy part.",
    "Built to run on your machine.",
    "Local when you want it.",
    "What should we figure out?",
    "One prompt is enough to begin.",
    "Turn a rough idea into a plan.",
    "Ask for the short version or the deep one.",
    "Drop in code, a file, or a question.",
    "Let's get the first draft down.",
    "What would make today easier?",
    "Start with the part that's stuck.",
    "Plain answers, no detours.",
    "What are we improving?",
    "Think out loud; I'll keep up.",
    "A good place to start is the problem.",
    "Bring the context; we'll find the thread.",
    "Ask for code, a plan, or an explanation.",
    "The model is ready when you are.",
    "What's the next small step?",
    "Pick up a saved prompt, or start clean.",
  ];
  function samplePills() {
    const copy = [...SUGGESTIONS];
    for (let i = copy.length - 1; i > 0; i -= 1) {
      const j = Math.floor(Math.random() * (i + 1));
      [copy[i], copy[j]] = [copy[j], copy[i]];
    }
    return copy.slice(0, 3);
  }
  let pills = samplePills();
  let recentLimit = RECENT_INITIAL;
  const BOOTED_AT = Date.now();
  const PROVIDERS = [
    ["local", "Local (Dizel)", "Dizel.png"],
    ["groq", "Groq", "Groq.png"],
    ["openai", "OpenAI", "chatgpt.png"],
    ["anthropic", "Anthropic", "claude.png"],
    ["gemini", "Gemini", "gemini.png"],
    ["xai", "xAI", "xai.png"],
    ["mistral", "Mistral", "mistral-ai.png"],
    ["cohere", "Cohere", "cohere.png"],
    ["ollama", "Ollama", "ollama.png"],
    ["ai21", "AI21", "ai21-labs.png"],
    ["azure", "Azure", "microsoft-azure-openaI.png"],
    ["meta", "Meta", "meta.png"],
  ];

  const $ = (sel) => document.querySelector(sel);
  const openThoughts = new Set();

  function formatThoughtClock(ms) {
    const total = Math.max(0, Math.floor(ms / 1000));
    if (total < 60) return `${total}s`;
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    if (hours > 0) return `${hours}h ${minutes}m`;
    return `${minutes}m ${total % 60}s`;
  }

  /* ── Icons ──────────────────────────────────────────────────────────────
     UI glyphs are inline paths from desktop/src/components/Icon.tsx. The
     sidebar and workspace glyphs come from desktop/public/icons, generated
     into js/demo-icons.js by docs/tools/build-icons.py. Nothing is loaded as
     a CSS mask: file:// blocks external SVG masks. */
  const PATHS = {
    plus: '<path d="M12 5v14"/><path d="M5 12h14"/>',
    close: '<path d="M18 6L6 18"/><path d="M6 6l12 12"/>',
    check: '<path d="M20 6L9 17l-5-5"/>',
    copy: '<rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>',
    refresh: '<path d="M21 12a9 9 0 1 1-2.64-6.36"/><path d="M21 3v6h-6"/>',
    restore: '<path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v5h5"/>',
    arrowUp: '<path d="M12 19V5"/><path d="M5 12l7-7 7 7"/>',
    pin: '<path d="M9 3h6v5.2l3 3V13H6v-1.8l3-3V3z"/><path d="M12 13v8"/>',
    trash: '<path d="M3 6h18"/><path d="M8 6V4h8v2"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v5M14 11v5"/>',
    archive: '<rect x="3" y="4" width="18" height="4" rx="1"/><path d="M5 8v11a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V8"/><path d="M10 12h4"/>',
    search: '<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/>',
    code: '<path d="M16 18l6-6-6-6"/><path d="M8 6l-6 6 6 6"/>',
    book: '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>',
    globe: '<circle cx="12" cy="12" r="10"/><path d="M2 12h20"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>',
    alert: '<path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    sparkles: '<path d="M12 3l1.9 4.6L18.5 9.5 13.9 11.4 12 16l-1.9-4.6L5.5 9.5l4.6-1.9z"/><path d="M19 15l.9 2.1L22 18l-2.1.9L19 21l-.9-2.1L16 18l2.1-.9z"/>',
    file: '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/>',
    folder: '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    image: '<rect x="3" y="3" width="18" height="18" rx="2.5"/><circle cx="8.5" cy="8.5" r="1.6"/><path d="M21 15.5l-4.5-4.5L6 21"/>',
    chevronUp: '<path d="M18 15l-6-6-6 6"/>',
    chevronDown: '<path d="M6 9l6 6 6-6"/>',
    chevronRight: '<path d="M9 18l6-6-6-6"/>',
  };

  function icon(name, size) {
    const s = size || 14;
    return `<svg width="${s}" height="${s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${PATHS[name] || ""}</svg>`;
  }

  const DRAW = window.DIZEL_DRAW || {};

  function drawIcon(name, size) {
    const d = DRAW[name];
    if (!d) return "";
    if (d.fill) {
      return `<svg width="${size}" height="${size}" viewBox="${d.vb}" fill="currentColor" stroke="none" aria-hidden="true">${d.body}</svg>`;
    }
    return `<svg width="${size}" height="${size}" viewBox="${d.vb}" fill="none" stroke="currentColor" stroke-width="${d.sw}" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${d.body}</svg>`;
  }

  /* ── Persistence ───────────────────────────────────────────────────────── */
  const seed = () => {
    const now = Date.now();
    return {
      theme: "dark",
      tier: "Flash",
      mode: "Fast",
      enterToSend: true,
      username: "Daniel",
      systemPrompt: "You are Dizel, a capable assistant. Be direct, use Markdown when it helps.",
      provider: "local",
      collapsed: false,
      showStatusbar: true,
      showRecent: true,
      view: null,
      activeId: null,
      speech: Object.assign({}, SPEECH_DEFAULTS),
      prompts: [
        {
          id: "prompt-review",
          title: "Code review",
          body: "Review this code for correctness first, then for clarity. Point out bugs before style. Suggest the smallest fix.",
        },
        {
          id: "prompt-explain",
          title: "Explain simply",
          body: "Explain this to a smart beginner. Use one concrete example and no jargon.",
        },
      ],
      sessions: [
        {
          id: "seed-identity",
          title: "What is your name?",
          preview: "I am Dizel, an AI assistant created by the Dizel team.",
          pinned: false,
          archived: false,
          ts: now - 6e5,
          messages: [
            { role: "user", content: "What is your name?" },
            {
              role: "assistant",
              content:
                "I am Dizel, an AI assistant created by the Dizel team.\n\nThis window is the desktop app's shell; the browser demo scripts the replies so the page can run without a model.",
            },
          ],
        },
        {
          id: "seed-variable",
          title: "Explain what a variable is",
          preview: "A variable is a named box in memory whose contents can change.",
          pinned: false,
          archived: false,
          ts: now - 3.6e6,
          messages: [
            { role: "user", content: "Explain what a variable is in one sentence." },
            {
              role: "assistant",
              content:
                "A variable is a named box in memory whose contents can change while the program runs.\n\n```python\ncount = 0      # the box is named count\ncount = count + 1   # its contents change\n```",
            },
          ],
        },
        {
          id: "seed-reverse",
          title: "Write a function that reverses a string",
          preview: "def reverse(text): return text[::-1]",
          pinned: true,
          archived: false,
          ts: now - 8.6e6,
          messages: [
            { role: "user", content: "Write a function that reverses a string." },
            {
              role: "assistant",
              content:
                '```python\ndef reverse(text: str) -> str:\n    """Return text with its characters in reverse order."""\n    return text[::-1]\n\nprint(reverse("dizel"))  # lezid\n```\n\nThe slice `[::-1]` walks the string backwards, so this stays O(n) and allocates one result string.',
              thinking:
                "The user wants a function that reverses a string.\n\nPython slicing accepts a step, so `text[::-1]` walks from the end to the start.\n\nPlan:\n- Keep it to one expression.\n- Mention the O(n) cost.\n- Note that the empty string needs no special case.",
              thoughtMs: 19000,
            },
          ],
        },
        {
          id: "seed-hi",
          title: "Hi",
          preview: "Hey. What are we working on?",
          pinned: false,
          archived: false,
          ts: now - 1.7e7,
          messages: [
            { role: "user", content: "Hi" },
            { role: "assistant", content: "Hey. What are we working on?" },
          ],
        },
        {
          id: "seed-summarise",
          title: "Summarise this README",
          preview: "A short overview of the project, its setup and its limits.",
          pinned: false,
          archived: false,
          ts: now - 2.2e7,
          messages: [
            { role: "user", content: "Summarise this README in five bullets." },
            {
              role: "assistant",
              content:
                "Here is the short version:\n\n- Local-first chat app\n- Runs a small model on CPU\n- Desktop, terminal and API surfaces\n- No account required\n- Experimental preview",
            },
          ],
        },
        {
          id: "seed-failing-test",
          title: "Why does this test fail?",
          preview: "The fixture is created after the assertion runs.",
          pinned: false,
          archived: false,
          ts: now - 2.8e7,
          messages: [
            { role: "user", content: "Why does this test fail?" },
            {
              role: "assistant",
              content:
                "The fixture is created after the assertion runs, so the value is still `None` when it is checked. Move the fixture setup above the assertion.",
            },
          ],
        },
        {
          id: "seed-translate",
          title: "Translate a paragraph",
          preview: "Shorter sentences and a plain register.",
          pinned: false,
          archived: false,
          ts: now - 3.4e7,
          messages: [
            { role: "user", content: "Translate this paragraph to English." },
            {
              role: "assistant",
              content:
                "Here is the translation, kept close to the original meaning:\n\n> The tool stays on this machine. Nothing is uploaded while the model is running.",
            },
          ],
        },
        {
          id: "seed-cli-plan",
          title: "Plan a small CLI tool",
          preview: "Start with one command, then add flags.",
          pinned: false,
          archived: false,
          ts: now - 4.2e7,
          messages: [
            { role: "user", content: "Plan a small CLI tool for converting files." },
            {
              role: "assistant",
              content:
                "1. Pick the input and output formats.\n2. Build one `convert` command.\n3. Add flags only when a second use case appears.\n4. Test with one real file per format.",
            },
          ],
        },
        {
          id: "seed-archived",
          title: "Old notes on tokenizers",
          preview: "Byte-pair encoding merges the most frequent pair first.",
          pinned: false,
          archived: true,
          ts: now - 2.6e7,
          messages: [
            { role: "user", content: "Summarise byte-pair encoding." },
            {
              role: "assistant",
              content:
                "Byte-pair encoding starts from characters, counts adjacent pairs, and merges the most frequent pair into a new token. Repeat until the vocabulary is full.",
            },
          ],
        },
      ],
    };
  };

  function normalize(next) {
    const base = seed();
    const merged = Object.assign(base, next || {});
    if (!Array.isArray(merged.sessions) || merged.sessions.length === 0) {
      merged.sessions = base.sessions;
    }
    merged.sessions = merged.sessions.map((session) =>
      Object.assign(
        { pinned: false, archived: false, preview: "", messages: [] },
        session,
        { messages: Array.isArray(session.messages) ? session.messages : [] },
      ),
    );
    if (!Array.isArray(merged.prompts)) merged.prompts = base.prompts;
    merged.speech = Object.assign(
      {},
      SPEECH_DEFAULTS,
      merged.speech && typeof merged.speech === "object" ? merged.speech : {},
    );
    if (!SPEECH_MODELS.some(([value]) => value === merged.speech.model)) {
      merged.speech.model = SPEECH_DEFAULTS.model;
    }
    if (!SPEECH_LANGUAGES.some(([value]) => value === merged.speech.language)) {
      merged.speech.language = SPEECH_DEFAULTS.language;
    }
    if (!SPEECH_TASKS.some(([value]) => value === merged.speech.task)) {
      merged.speech.task = SPEECH_DEFAULTS.task;
    }
    merged.speech.prompt = String(merged.speech.prompt || "").slice(0, 400);
    if (!TIERS.includes(merged.tier)) merged.tier = "Flash";
    if (!MODES.includes(merged.mode)) merged.mode = "Fast";
    if (!THEMES.some(([value]) => value === merged.theme)) merged.theme = "dark";
    if (merged.view && !VIEW_TITLES[merged.view]) merged.view = null;
    const activeSession = merged.sessions.find((session) => session.id === merged.activeId);
    if (activeSession && activeSession.archived) merged.activeId = null;
    if (merged.activeId !== null && !activeSession) merged.activeId = null;
    delete merged.boot;
    return merged;
  }

  let state;
  try {
    state = normalize(JSON.parse(localStorage.getItem(KEY)));
  } catch (_) {
    state = seed();
  }
  state.boot = "starting";

  // #light / #dark-blue let the page open straight into a theme (used for docs).
  const hashTheme = location.hash.replace("#", "");
  if (THEMES.some(([value]) => value === hashTheme)) state.theme = hashTheme;

  const save = () => {
    try {
      const copy = Object.assign({}, state);
      delete copy.boot;
      localStorage.setItem(KEY, JSON.stringify(copy));
    } catch (_) {
      /* private mode — the demo just won't persist */
    }
  };

  const active = () => state.sessions.find((session) => session.id === state.activeId) || null;

  const previewOf = (session) => {
    if (session.preview) return session.preview;
    const last = session.messages[session.messages.length - 1];
    if (!last) return "No messages yet";
    return last.content.replace(/[#*`>\n]+/g, " ").replace(/\s+/g, " ").trim().slice(0, 90);
  };

  const bucket = (ts) => {
    const now = new Date();
    const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
    if (ts >= startOfToday) return "Today";
    if (ts >= startOfToday - 86400000) return "Yesterday";
    if (ts >= startOfToday - 6 * 86400000) return "This week";
    return "Older";
  };

  const GROUP_ORDER = ["Pinned", "Today", "Yesterday", "This week", "Older"];

  /* ── Markdown (escaped, then a tiny subset rendered) ───────────────────── */
  const esc = (value) =>
    String(value).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const escAttr = (value) => esc(value).replace(/"/g, "&quot;").replace(/'/g, "&#39;");

  function safeUrl(url, image) {
    const pattern = image
      ? /^(?:https?:\/\/|\.{0,2}\/|assets\/)/i
      : /^(?:https?:\/\/|mailto:|#|\.{0,2}\/|assets\/)/i;
    return pattern.test(url) ? url : "#";
  }

  const attr = (value) => String(value).replace(/"/g, "&quot;");

  function inline(value) {
    const codes = [];
    let text = value.replace(/`([^`]+)`/g, (_, code) => {
      codes.push(`<code>${code}</code>`);
      return `\u0000${codes.length - 1}\u0000`;
    });

    text = text
      .replace(
        /!\[([^\]]*)\]\(([^)\s]+)\)/g,
        (_, alt, src) => `<img src="${attr(safeUrl(src, true))}" alt="${attr(alt)}">`,
      )
      .replace(
        /\[([^\]]+)\]\(([^)\s]+)\)/g,
        (_, label, href) =>
          `<a href="${attr(safeUrl(href, false))}" target="_blank" rel="noreferrer">${label}</a>`,
      )
      .replace(/~~([^~]+)~~/g, "<del>$1</del>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^\w])__([^_]+)__(?!\w)/g, "$1<strong>$2</strong>")
      .replace(/\*([^*]+)\*/g, "<em>$1</em>")
      .replace(/(^|[^\w])_([^_]+)_(?!\w)/g, "$1<em>$2</em>");

    return text.replace(/\u0000(\d+)\u0000/g, (_, index) => codes[Number(index)]);
  }

  const isTableRow = (line) => {
    const text = (line || "").trim();
    return text.startsWith("|") && text.endsWith("|") && text.length > 2;
  };

  const tableCells = (line) =>
    line
      .trim()
      .replace(/^\||\|$/g, "")
      .split("|")
      .map((cell) => cell.trim());

  const isTableDivider = (line) => {
    if (!isTableRow(line)) return false;
    const cells = tableCells(line);
    return cells.length > 0 && cells.every((cell) => /^:?-+:?$/.test(cell));
  };

  function mdToHtml(src) {
    const lines = String(src).split("\n");
    const out = [];
    let list = null;
    let para = [];
    let code = null;
    let quote = null;

    const flushPara = () => {
      if (para.length) {
        out.push(`<p>${inline(esc(para.join(" ")))}</p>`);
        para = [];
      }
    };
    const flushList = () => {
      if (list) {
        const items = out.pop() || "";
        out.push(`<${list}>${items}</${list}>`);
        list = null;
      }
    };
    const flushQuote = () => {
      if (quote) {
        out.push(`<blockquote>${mdToHtml(quote.join("\n"))}</blockquote>`);
        quote = null;
      }
    };

    const tableRow = (cells, headers, aligns, tag) =>
      `<tr>${headers
        .map((_, index) => {
          const align = aligns[index];
          const style =
            align === "center" || align === "right" ? ` style="text-align:${align}"` : "";
          return `<${tag}${style}>${inline(esc(cells[index] || ""))}</${tag}>`;
        })
        .join("")}</tr>`;

    for (let i = 0; i < lines.length; i += 1) {
      const raw = lines[i];
      const line = raw.replace(/\s+$/, "");

      if (code !== null) {
        if (line.startsWith("```")) {
          out.push(`<pre><code>${esc(code.join("\n"))}</code></pre>`);
          code = null;
        } else {
          code.push(raw);
        }
        continue;
      }

      if (!line.trimStart().startsWith(">")) flushQuote();

      if (line.startsWith("```")) {
        flushPara();
        flushList();
        code = [];
        continue;
      }

      if (!line.trim()) {
        flushPara();
        flushList();
        continue;
      }

      if (isTableRow(line) && isTableDivider(lines[i + 1])) {
        flushPara();
        flushList();
        const headers = tableCells(line);
        const aligns = tableCells(lines[i + 1]).map((cell) => {
          const left = cell.startsWith(":");
          const right = cell.endsWith(":");
          return left && right ? "center" : right ? "right" : left ? "left" : "";
        });
        i += 1;
        const rows = [];
        while (i + 1 < lines.length && isTableRow(lines[i + 1])) {
          rows.push(tableCells(lines[i + 1]));
          i += 1;
        }
        out.push(
          `<table><thead>${tableRow(headers, headers, aligns, "th")}</thead><tbody>${rows
            .map((row) => tableRow(row, headers, aligns, "td"))
            .join("")}</tbody></table>`,
        );
        continue;
      }

      if (/^\s*(?:---+|\*\*\*+|___+)\s*$/.test(line)) {
        flushPara();
        flushList();
        out.push("<hr>");
        continue;
      }

      if (/^\s*>/.test(line)) {
        flushPara();
        flushList();
        if (!quote) quote = [];
        quote.push(line.replace(/^\s*>\s?/, ""));
        continue;
      }

      const heading = line.match(/^(#{1,6})\s+(.*)$/);
      if (heading) {
        flushPara();
        flushList();
        const level = heading[1].length;
        const text = heading[2].replace(/\s+#+\s*$/, "");
        out.push(`<h${level}>${inline(esc(text))}</h${level}>`);
        continue;
      }

      const bullet = line.match(/^\s*[-*]\s+(.*)$/);
      const numbered = line.match(/^\s*\d+\.\s+(.*)$/);
      if (bullet || numbered) {
        flushPara();
        const kind = bullet ? "ul" : "ol";
        if (list !== kind) {
          flushList();
          out.push("");
          list = kind;
        }
        let text = (bullet || numbered)[1];
        let task = null;
        if (bullet) {
          const checked = text.match(/^\[([ xX])\]\s+(.*)$/);
          if (checked) {
            task = checked[1].toLowerCase() === "x";
            text = checked[2];
          }
        }
        const marker =
          task === null
            ? ""
            : `<input type="checkbox" disabled${task ? " checked" : ""}> `;
        out[out.length - 1] =
          (out[out.length - 1] || "") +
          `<li${task === null ? "" : ' class="task"'}>${marker}${inline(esc(text))}</li>`;
        continue;
      }

      flushList();
      para.push(line);
    }

    if (code !== null) out.push(`<pre><code>${esc(code.join("\n"))}</code></pre>`);
    flushPara();
    flushList();
    flushQuote();
    return out.join("");
  }

  /* ── Scripted replies ──────────────────────────────────────────────────── */
  function thoughtFor(prompt) {
    const p = prompt.toLowerCase();
    if (!/reverse|reverses|backwards/.test(p)) return "";
    return [
      "The user wants a function that reverses a string.",
      "Python slicing accepts a step, so `text[::-1]` walks from the end to the start.",
      "Plan:",
      "- Keep it to one expression.",
      "- Mention the O(n) cost.",
      "- Note that the empty string needs no special case.",
    ].join("\n\n");
  }

  function reply(prompt) {
    const p = prompt.toLowerCase();
    const mode = state.mode;

    if (/^(hi|hey|hello|yo)\b/.test(p) && p.length < 24) {
      return "Hey. Ask me to explain something, write a small function, or sketch a plan. I keep it short unless you want depth.";
    }

    if (/(your name|who are you|who made|created you|are you (gpt|chatgpt|qwen))/.test(p)) {
      return [
        "I am Dizel, an AI assistant created by the Dizel team.",
        "",
        "This is the browser demo, so the reply you are reading is scripted. The desktop app runs the real checkpoint locally.",
      ].join("\n");
    }

    if (/\breverses?\b/.test(p)) {
      return [
        "```python",
        "def reverse(text: str) -> str:",
        '    """Return text with its characters in reverse order."""',
        "    return text[::-1]",
        "",
        'print(reverse("dizel"))  # lezid',
        "```",
        "",
        "The slice `[::-1]` walks the string backwards, so this stays O(n) and allocates one result string.",
      ].join("\n");
    }

    if (/(function|code|python|javascript|sql|program|script)/.test(p) || mode === "Coding") {
      return [
        "Here is the smallest version that stays readable:",
        "",
        "```python",
        "def chunk(items: list, size: int) -> list[list]:",
        '    """Split items into consecutive chunks of at most `size`."""',
        "    if size < 1:",
        '        raise ValueError("size must be positive")',
        "    return [items[i:i + size] for i in range(0, len(items), size)]",
        "",
        'print(chunk([1, 2, 3, 4, 5], 2))  # [[1, 2], [3, 4], [5]]',
        "```",
        "",
        "The slice does the work. The guard is there because `range(0, n, 0)` raises instead of doing something sensible.",
      ].join("\n");
    }

    if (/(plan|steps|roadmap|approach|how do i)/.test(p) || mode === "Planning") {
      return [
        "A workable order for this:",
        "",
        "1. **Write the smallest failing case**: one input, one expected output.",
        "2. **Make it pass** with the simplest code you can defend.",
        "3. **Add the awkward cases**: empty input, one item, a size larger than the list.",
        "4. **Only then** refactor, with the tests as the safety net.",
        "",
        "If a step takes more than an hour, it is really two steps.",
      ].join("\n");
    }

    if (mode === "Deep Think") {
      return [
        "## Working it through",
        "",
        "1. **What is actually being asked** — " + prompt.trim().slice(0, 140),
        "2. **Constraints that shape the answer** — keep the response useful, not just long.",
        "3. **The main trade-off** — depth buys confidence, but it costs time and context.",
        "4. **Failure mode to watch** — a polished answer can still hide a weak assumption.",
        "",
        "### Answer",
        "",
        "Start with the smallest correct version, then add depth only where the question earns it. Check the assumption above before building on it.",
      ].join("\n");
    }

    if (/(explain|what is|what are|how does|why)/.test(p) || mode === "Thinking") {
      return [
        "Short version: a variable is a **name bound to a value** that the program can change later.",
        "",
        "- The name lives in a scope; the value lives in memory",
        "- Rebinding replaces the value, it does not edit it in place",
        "- Two names can point at the same object, which is why mutation surprises people",
        "",
        "> If you remember one thing: the box is the name, not the value.",
      ].join("\n");
    }

    return [
      "Here is how I would read that:",
      "",
      "- **What you are asking**: " + prompt.trim().slice(0, 120),
      "- **What matters first**: the constraint that makes the other choices easy",
      "- **What I would do next**: a small step you can check before committing",
      "",
      "Tell me the constraint and I will narrow it down.",
    ].join("\n");
  }

  /* ── DOM ───────────────────────────────────────────────────────────────── */
  const el = {
    main: $("#main"),
    view: $("#view"),
    composerWrap: $("#composer-wrap"),
    statusbarWrap: $("#statusbar-wrap"),
    list: $("#sb-list"),
    recent: $("#sb-recent"),
    spacer: $("#sidebar-spacer"),
    search: $("#search"),
    sbSearch: $("#sb-search"),
    sidebar: $("#sidebar"),
    subsidebar: $("#subsidebar"),
    statusPill: $("#status-pill"),
    loadbar: $("#loadbar"),
    statusbar: $("#statusbar"),
    composer: $("#composer"),
    input: $("#input"),
    send: $("#btn-send"),
    popover: $("#popover"),
    settings: $("#settings"),
    settingsTabs: $("#settings-tabs"),
    settingsPanel: $("#settings-panel"),
    profile: $("#profile"),
  };

  let streaming = null;
  let loaderTimer = null;
  let thoughtTimer = null;
  let thoughtTimeout = null;
  let thinkingMessage = null;
  let confirmId = null;
  let confirmTimer = null;
  let popoverAnchor = null;
  let subQuery = "";
  let subOpenForm = false;

  /* ── Composer placement ──────────────────────────────────────────────────
     ChatView renders the composer inside the welcome hero; App renders it
     docked above the status bar. One node, moved between the two slots, so
     the draft and focus survive the transition. */
  function parkComposer() {
    if (el.composerWrap.parentElement !== el.main) {
      el.main.insertBefore(el.composerWrap, el.statusbarWrap);
    }
  }

  function placeComposer() {
    const slot = $("#welcome-composer-slot");
    if (slot) slot.appendChild(el.composerWrap);
    else parkComposer();
  }

  /* ── Rendering ─────────────────────────────────────────────────────────── */
  function dayPart(date) {
    const hour = (date || new Date()).getHours();
    if (hour < 12) return "morning";
    if (hour < 18) return "afternoon";
    return "evening";
  }

  function greeting(date) {
    return `Good ${dayPart(date)}`;
  }

  function pick(list) {
    return list[Math.floor(Math.random() * list.length)];
  }

  function sampleWelcome(part) {
    return { headline: pick(HEADLINES[part]), subline: pick(SUBLINES) };
  }

  function fillWelcome(template, values) {
    return template
      .replace(/\{greeting\}/g, values.greeting)
      .replace(/\{name\}/g, values.name);
  }

  let welcomePart = dayPart();
  let welcomeCopy = sampleWelcome(welcomePart);

  function typingHtml() {
    return '<div class="typing"><span></span><span></span><span></span></div>';
  }

  function loaderHtml() {
    const stage = state.boot === "starting" ? "Starting…" : "Reading GGUF metadata";
    return `<div class="loader">
      <div class="loader-head">
        <span class="loader-spinner" aria-hidden="true"></span>
        <h1>Loading ${MODEL.name}</h1>
      </div>
      <p class="loader-stage">${stage}</p>
      <div class="loader-track" aria-hidden="true"><span></span></div>
      <span class="loader-elapsed" id="loader-elapsed">0s</span>
    </div>`;
  }

  function sidebarRow(session) {
    const isActive = session.id === state.activeId;
    return `<div class="sb-row ${isActive ? "active" : ""} ${session.id === confirmId ? "confirming" : ""}" data-id="${escAttr(session.id)}">
      <button class="sb-row-label" data-act="open" title="${escAttr(session.title)}">${esc(session.title)}</button>
      <div class="sb-row-actions">
        <button class="row-act" data-act="pin" title="${session.pinned ? "Unpin" : "Pin"}" aria-label="${session.pinned ? "Unpin" : "Pin"}">${icon("pin", 12)}</button>
        <button class="row-act" data-act="archive" title="Archive" aria-label="Archive">${icon("archive", 12)}</button>
        <button class="row-act danger" data-act="delete" title="Delete" aria-label="Delete">${icon("trash", 12)}</button>
      </div>
    </div>`;
  }

  function renderSidebar() {
    const live = state.sessions.filter((session) => !session.archived);
    const showSearch = live.length > 6 && !state.collapsed && state.showRecent;
    el.sbSearch.hidden = !showSearch;

    const query = showSearch ? (el.search.value || "").trim().toLowerCase() : "";
    const visible = live.filter(
      (session) =>
        !query ||
        `${session.title} ${previewOf(session)} ${session.messages.map((m) => m.content).join(" ")}`
          .toLowerCase()
          .includes(query),
    );

    const pinned = visible.filter((session) => session.pinned);
    const rest = visible.filter((session) => !session.pinned);
    const searching = query.length > 0;
    const shownRest = searching ? rest : rest.slice(0, recentLimit);
    const hidden = rest.length - shownRest.length;

    el.recent.classList.toggle("is-open", state.showRecent);
    el.spacer.hidden = !(state.collapsed || !state.showRecent);

    const toggle = $("#toggle-recent");
    toggle.innerHTML = icon(state.showRecent ? "chevronUp" : "chevronDown", 12);
    toggle.setAttribute("aria-label", state.showRecent ? "Hide recent chats" : "Show recent chats");
    toggle.setAttribute("title", state.showRecent ? "Hide recent chats" : "Show recent chats");

    let html = "";
    if (pinned.length) {
      html += `<div class="sb-sub">Pinned</div>` + pinned.map(sidebarRow).join("");
      if (rest.length) html += `<div class="sb-sub">All chats</div>`;
    }
    html += shownRest.map(sidebarRow).join("");
    if (!searching && hidden > 0 && recentLimit < RECENT_MAX) {
      html += `<button class="sb-more" id="sb-more">Show more ${icon("chevronDown", 12)}</button>`;
    } else if (!searching && hidden > 0) {
      html += `<div class="sb-search-hint">${icon("search", 12)}<span>Showing ${RECENT_MAX} most recent. Use search for older chats.</span></div>`;
    }
    if (!visible.length) {
      html += `<div class="sb-empty">${query ? "No matches" : "No chats yet"}</div>`;
    }
    el.list.innerHTML = html;

    const more = $("#sb-more");
    if (more) {
      more.onclick = () => {
        recentLimit = Math.min(RECENT_MAX, recentLimit + RECENT_PAGE);
        renderSidebar();
      };
    }
  }

  function messageHtml(message, index, sessionId) {
    if (message.role === "user") {
      return `<div class="msg user"><div class="user-card">
        <div class="msg-label">You</div>
        <div class="user-text">${esc(message.content)}</div>
      </div></div>`;
    }

    const live = Boolean(message.streaming);
    const thinking = message.thinking || "";
    const isThinking = live && !message.content && message.thoughtMs === undefined;
    const thinkKey = `${sessionId}:${index}`;
    const thinkOpen = openThoughts.has(thinkKey);
    let thoughtLabel = "";
    if (thinking) {
      if (message.thoughtMs !== undefined) {
        thoughtLabel = `Thought for ${formatThoughtClock(message.thoughtMs)}`;
      } else if (message.thoughtStartedAt !== undefined) {
        thoughtLabel = `Thinking for ${formatThoughtClock(Date.now() - message.thoughtStartedAt)}`;
      } else {
        thoughtLabel = "Thought";
      }
    }
    const think = thinking
      ? `<div class="think${thinkOpen ? " is-open" : ""}${isThinking ? " is-thinking" : ""}">
          <button type="button" class="think-head" data-think="${index}" aria-expanded="${thinkOpen}">
            ${icon("sparkles", 14)}
            <span class="think-label">${thoughtLabel}</span>
            ${icon("chevronDown", 12)}
          </button>
          <div class="think-panel" aria-hidden="${!thinkOpen}">
            <div class="think-panel-inner">
              <div class="think-body"><div class="md">${mdToHtml(thinking)}</div></div>
            </div>
          </div>
        </div>`
      : "";

    const body =
      thinking && live && !message.content
        ? ""
        : live && !message.content
          ? typingHtml()
          : mdToHtml(message.content) + (live ? '<span class="caret"></span>' : "");

    const actions = live
      ? ""
      : `<div class="msg-actions">
          <button class="act" data-copy="${index}">${icon("copy", 12)} Copy</button>
          <button class="act" data-regen="${index}">${icon("refresh", 12)} Regenerate</button>
          <span class="spacer"></span>
          <span class="msg-meta">${message.content.length.toLocaleString()} chars · scripted</span>
        </div>`;

    return `<div class="msg assistant" data-msg="${index}">
      <div class="asst-head ${live ? "is-live" : ""}">
        <img class="logo-mark" src="assets/demo/Dizel.png" alt="">
        <span class="asst-name">Dizel</span>
      </div>
      ${think}
      <div class="md" data-md>${body}</div>
      ${actions}
    </div>`;
  }

  function recentHtml() {
    const recent = state.sessions
      .filter((session) => !session.archived && session.messages.length && session.id !== state.activeId)
      .sort((a, b) => b.ts - a.ts)
      .slice(0, 3);

    if (!recent.length) return "";

    return `<div class="recent-block">
      <div class="recent-head">Pick up where you left off</div>
      ${recent
        .map(
          (session) => `<button class="recent-row" data-session="${escAttr(session.id)}" title="${escAttr(previewOf(session))}">
            <span class="recent-name">${esc(session.title)}</span>
            <span class="recent-sum">${esc(previewOf(session))}</span>
          </button>`,
        )
        .join("")}
    </div>`;
  }

  function welcomeHtml() {
    const loading = state.boot !== "ready";
    const values = { greeting: greeting(), name: state.username };
    const hero = loading
      ? loaderHtml()
      : `<h1>${esc(fillWelcome(welcomeCopy.headline, values))}</h1><p>${esc(
          fillWelcome(welcomeCopy.subline, values),
        )}</p>`;

    return `<div class="welcome ${loading ? "is-loading" : ""}">
      <div class="welcome-hero">
        ${hero}
        <div class="welcome-composer" id="welcome-composer-slot"></div>
        <div class="action-row">
          ${pills
            .map(
              (item) =>
                `<button class="action-pill" data-prompt="${escAttr(item.prompt)}">${esc(item.label)} ${icon(item.icon, 14)}</button>`,
            )
            .join("")}
        </div>
      </div>
      ${recentHtml()}
    </div>`;
  }

  function renderView() {
    parkComposer();
    const session = active();

    if (!session || !session.messages.length) {
      el.view.innerHTML = welcomeHtml();
      placeComposer();

      el.view.querySelectorAll("[data-prompt]").forEach((button) => {
        button.onclick = () => {
          el.input.value = button.dataset.prompt;
          el.input.focus();
          resizeInput();
          syncSend();
        };
      });

      el.view.querySelectorAll("[data-session]").forEach((button) => {
        button.onclick = () => openSession(button.dataset.session);
      });
      return;
    }

    el.view.innerHTML = `<div class="stream-wrap"><div class="stream" id="stream">${session.messages
      .map((message, index) => messageHtml(message, index, session.id))
      .join("")}</div></div>`;
    const stream = $("#stream");
    if (stream) stream.scrollTop = stream.scrollHeight;

    el.view.querySelectorAll("[data-copy]").forEach((button) => {
      button.onclick = async () => {
        const message = session.messages[Number(button.dataset.copy)];
        try {
          await navigator.clipboard.writeText(message ? message.content : "");
          button.innerHTML = `${icon("check", 12)} Copied`;
          setTimeout(() => (button.innerHTML = `${icon("copy", 12)} Copy`), 1200);
        } catch (_) {
          button.textContent = "Copy failed";
        }
      };
    });

    el.view.querySelectorAll("[data-regen]").forEach((button) => {
      button.onclick = () => regenerate(Number(button.dataset.regen));
    });

    el.view.querySelectorAll("[data-think]").forEach((button) => {
      button.onclick = () => {
        const index = Number(button.dataset.think);
        const key = `${session.id}:${index}`;
        const open = !openThoughts.has(key);
        if (open) openThoughts.add(key);
        else openThoughts.delete(key);
        const block = button.closest(".think");
        block?.classList.toggle("is-open", open);
        button.setAttribute("aria-expanded", String(open));
        block?.querySelector(".think-panel")?.setAttribute("aria-hidden", String(!open));
      };
    });
  }

  function renderStatus() {
    const ready = state.boot === "ready";
    const loading = state.boot !== "ready";

    el.loadbar.classList.toggle("is-loading", loading);
    el.statusPill.classList.toggle("is-loading", loading);
    el.statusPill.innerHTML = loading
      ? `<span class="pill-spinner"></span><span class="pill-text">${
          state.boot === "starting" ? "Starting Dizel…" : "Loading model… reading GGUF metadata"
        }</span>`
      : `<span class="pill-text">${MODEL.name} · ${MODEL.params} · ${MODEL.dims} · ${MODEL.device}</span>`;

    const generating = Boolean(streaming);
    $("#sb-dot").className = "dot " + (generating || loading ? "busy" : ready ? "ok" : "");
    $("#sb-state").textContent = generating ? "Generating" : loading ? "Loading" : "Ready";
    $("#sb-model").textContent = MODEL.name;
    $("#sb-params").textContent = MODEL.params;
    $("#sb-device").textContent = MODEL.device;

    const session = active();
    const messages = session ? session.messages : [];
    const used = Math.round(messages.reduce((total, message) => total + message.content.length, 0) / 4);
    $("#sb-msgs").textContent = `${messages.length} msg`;
    $("#sb-ctx").textContent = `~${used.toLocaleString()} / ${CONTEXT_LEN} ctx`;
    $("#sb-used").textContent = `${Math.max(1, Math.ceil(used / 512))} / 1024 used`;
    const mins = Math.floor((Date.now() - BOOTED_AT) / 60000);
    $("#sb-time").textContent = `${Math.floor(mins / 60)}h ${mins % 60}m`;
    $("#ctx-fill").style.width = `${Math.min(100, Math.round((used / CONTEXT_LEN) * 100))}%`;

    el.input.disabled = loading;
    el.input.placeholder = loading ? "Waiting for the model..." : "Message Dizel...";

    // Same rule as App.tsx: the glow is for live-and-empty, not finished work.
    const liveTurn = Boolean(session && session.messages.some((message) => message.streaming));
    el.main.classList.toggle(
      "is-glowing",
      loading || Boolean(streaming) || liveTurn || !session || !session.messages.length,
    );
  }

  function syncSend() {
    const busy = Boolean(streaming);
    const ready = state.boot === "ready";
    el.send.disabled = busy ? false : !ready || !el.input.value.trim();
    el.send.innerHTML = busy
      ? '<span class="stop-square"></span>'
      : '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 19V5"/><path d="M5 12l7-7 7 7"/></svg>';
    el.send.setAttribute("aria-label", busy ? "Stop generating" : "Send");
    el.send.setAttribute("title", busy ? "Stop generating" : "Send");
  }

  function render() {
    document.documentElement.dataset.theme = state.theme;
    el.sidebar.classList.toggle("collapsed", state.collapsed);
    el.statusbarWrap.classList.toggle("is-open", state.showStatusbar);

    const statusToggle = $("#toggle-statusbar");
    statusToggle.style.transform = state.showStatusbar ? "" : "rotate(180deg)";
    statusToggle.setAttribute("aria-label", state.showStatusbar ? "Hide status bar" : "Show status bar");
    statusToggle.setAttribute("title", state.showStatusbar ? "Hide status bar" : "Show status bar");

    $("#model-label").textContent = MODEL.name;
    $("#tier-label").textContent = state.tier;
    const modeChip = $("#tier-mode");
    modeChip.hidden = state.mode === "Fast";
    modeChip.textContent = state.mode;
    $("#btn-tier").classList.toggle("is-armed", state.mode !== "Fast");

    $("#sb-avatar").textContent = state.username.slice(0, 1).toUpperCase();
    $("#sb-user").textContent = state.username;

    $("#nav-chats").classList.toggle("is-active", state.view === "Chats");
    $("#nav-workspaces").classList.toggle(
      "is-active",
      Boolean(state.view && state.view !== "Chats"),
    );

    renderStatus();
    renderSidebar();
    renderView();
    renderSubsidebar();
    syncSend();
    save();
  }

  /* ── Secondary sidebar ─────────────────────────────────────────────────── */
  function emptyHtml(name, title, body, hint) {
    return `<div class="sub-empty">
      <span class="sub-empty-icon">${icon(name, 18)}</span>
      <div class="sub-empty-title">${title}</div>
      <div class="sub-empty-body">${body}</div>
      ${hint ? `<div class="sub-empty-hint">${hint}</div>` : ""}
    </div>`;
  }

  function groupSessions(sessions) {
    const map = new Map();
    for (const session of sessions) {
      const key = session.pinned ? "Pinned" : bucket(session.ts);
      if (!map.has(key)) map.set(key, []);
      map.get(key).push(session);
    }
    return GROUP_ORDER.filter((label) => map.has(label)).map((label) => [label, map.get(label)]);
  }

  function subRow(session) {
    return `<button class="sub-row ${session.id === state.activeId ? "active" : ""}" data-session="${escAttr(session.id)}" title="${escAttr(session.title)}">${esc(session.title)}</button>`;
  }

  function chatsBody() {
    const query = subQuery.trim().toLowerCase();
    const sessions = state.sessions
      .filter((session) => !session.archived)
      .filter(
        (session) =>
          !query || `${session.title} ${previewOf(session)}`.toLowerCase().includes(query),
      )
      .sort((a, b) => b.ts - a.ts);

    const groups = groupSessions(sessions);
    const rows = groups
      .map(
        ([label, items]) =>
          `<div class="sb-sub">${label}</div>` + items.map(subRow).join(""),
      )
      .join("");

    return `<div class="sub-search">
        ${icon("search", 13)}
        <input id="sub-chat-search" type="text" placeholder="Search chats…" autocomplete="off" spellcheck="false" aria-label="Search chats" value="${escAttr(subQuery)}">
        <button class="sub-icon-btn" title="Import is desktop-only" aria-label="Import a chat (desktop only)" disabled>
          <span style="display:flex;transform:rotate(180deg)">${icon("arrowUp", 13)}</span>
        </button>
      </div>
      <div class="sub-list">
        ${rows || `<div class="sub-hint">No chats match.</div>`}
      </div>`;
  }

  function risetBody() {
    const needle = subQuery.trim().toLowerCase();
    const search = `<div class="sub-search">
        ${icon("search", 13)}
        <input id="sub-riset-search" type="text" placeholder="Search everything…" autocomplete="off" spellcheck="false" aria-label="Search everything" value="${escAttr(subQuery)}">
      </div>`;

    if (!needle) {
      return (
        search +
        emptyHtml(
          "search",
          "Global search",
          "Searches chat titles, previews and saved prompts. Full-text search over message bodies lands with the archive view.",
        )
      );
    }

    const chatHits = state.sessions.filter((session) =>
      `${session.title} ${previewOf(session)}`.toLowerCase().includes(needle),
    );
    const promptHits = state.prompts.filter((prompt) =>
      `${prompt.title} ${prompt.body}`.toLowerCase().includes(needle),
    );

    return (
      search +
      `<div class="sub-list">
        ${chatHits.length ? `<div class="sub-group">Chats · ${chatHits.length}</div>` : ""}
        ${chatHits.map(subRow).join("")}
        ${promptHits.length ? `<div class="sub-group">Prompts · ${promptHits.length}</div>` : ""}
        ${promptHits
          .map(
            (prompt) =>
              `<div class="sub-row static" title="${escAttr(prompt.body)}">${esc(prompt.title)}</div>`,
          )
          .join("")}
        ${
          chatHits.length + promptHits.length === 0
            ? `<div class="sub-hint">Nothing matched “${esc(subQuery)}”.</div>`
            : ""
        }
      </div>`
    );
  }

  function libraryBody() {
    const form = subOpenForm
      ? `<div class="sub-form">
          <input id="prompt-title" type="text" placeholder="Title" autocomplete="off" spellcheck="false" aria-label="Prompt title">
          <textarea id="prompt-body" rows="4" placeholder="Prompt text…" spellcheck="false" aria-label="Prompt text"></textarea>
          <div class="sub-form-actions">
            <button class="btn" id="prompt-cancel">Cancel</button>
            <button class="btn primary" id="prompt-save" disabled>Save</button>
          </div>
        </div>`
      : "";

    const cards = state.prompts
      .map(
        (prompt, index) => `<div class="prompt-gem" data-prompt-id="${escAttr(prompt.id)}">
          <button type="button" class="prompt-gem-main" data-prompt-use="${escAttr(prompt.id)}" title="${escAttr(prompt.body)}">
            <span class="prompt-gem-icon tone-${index % 4}">${icon("sparkles", 16)}</span>
            <span class="prompt-gem-title">${esc(prompt.title)}</span>
            <span class="prompt-gem-body">${esc(prompt.body)}</span>
            <span class="prompt-gem-use">Use ${icon("chevronRight", 11)}</span>
          </button>
          <div class="prompt-gem-actions">
            <button class="sub-icon-btn" data-prompt-copy="${escAttr(prompt.id)}" title="Copy prompt" aria-label="Copy ${escAttr(prompt.title)}">${icon("copy", 11)}</button>
            <button class="sub-icon-btn" data-prompt-del="${escAttr(prompt.id)}" title="Delete prompt" aria-label="Delete ${escAttr(prompt.title)}">${icon("close", 11)}</button>
          </div>
        </div>`,
      )
      .join("");

    return `<div class="sub-intro">Save your best prompts, instructions and persona structures here.</div>
      <button class="sub-primary" id="sub-new-prompt">${icon("plus", 13)} New Prompt</button>
      ${form}
      <div class="sub-group">Saved prompts</div>
      ${state.prompts.length ? `<div class="prompt-grid">${cards}</div>` : `<div class="sub-hint">None saved yet.</div>`}`;
  }

  function monitorBody() {
    const session = active();
    const messages = session ? session.messages : [];

    return `<div class="stat-card">
        <div class="stat-card-title">System Resources</div>
        <div class="stat-row"><span>CPU</span><b>—</b></div>
        <div class="meter" aria-hidden="true"><span data-tone="cool" style="width:0%"></span></div>
        <div class="stat-row"><span>RAM</span><b>—</b></div>
        <div class="meter" aria-hidden="true"><span data-tone="cool" style="width:0%"></span></div>
        <div class="sub-hint">System stats come from the desktop sidecar.</div>
      </div>
      <div class="stat-card">
        <div class="stat-card-title">Current Session</div>
        <div class="stat-row"><span>Provider</span><b>Local (Dizel)</b></div>
        <div class="stat-row"><span>Model</span><b>${MODEL.name}</b></div>
        <div class="stat-row"><span>Params</span><b>${MODEL.params}</b></div>
        <div class="stat-row"><span>Context</span><b>${CONTEXT_LEN.toLocaleString()} tokens</b></div>
        <div class="stat-row"><span>Device</span><b>${MODEL.device}</b></div>
        <div class="stat-row"><span>Messages</span><b>${messages.length}</b></div>
      </div>`;
  }

  function archivedBody() {
    const archived = state.sessions.filter((session) => session.archived);
    if (!archived.length) {
      return emptyHtml(
        "archive",
        "Nothing archived",
        "Archived chats leave the sidebar but stay on disk.",
        "Use a chat's ⋯ menu and choose Archive.",
      );
    }

    return `<div class="sub-intro">${archived.length} archived ${
      archived.length === 1 ? "chat" : "chats"
    }. Restoring returns them to the sidebar.</div>
      <div class="sub-list">
        ${archived
          .map(
            (session) => `<div class="sub-list-row">
              ${subRow(session)}
              <button class="sub-icon-btn" data-restore="${escAttr(session.id)}" title="Restore" aria-label="Restore">${icon("restore", 12)}</button>
            </div>`,
          )
          .join("")}
      </div>`;
  }

  function subBody() {
    switch (state.view) {
      case "Chats":
        return chatsBody();
      case "Riset":
        return risetBody();
      case "Library":
        return libraryBody();
      case "Monitor":
        return monitorBody();
      case "Archived":
        return archivedBody();
      case "Image":
        return emptyHtml(
          "image",
          "Image gallery",
          "Images you attach to a message will be collected here.",
          "Attachments are text-only for now. This fills in once vision lands.",
        );
      case "Presentation":
        return emptyHtml(
          "file",
          "Presentations",
          "Slide decks you generate will be listed here.",
          "Not implemented yet. The desktop app doesn't generate slides either.",
        );
      default:
        return "";
    }
  }

  function refocus(selector) {
    const node = $(selector);
    if (!node) return;
    node.focus();
    if (node.setSelectionRange) node.setSelectionRange(node.value.length, node.value.length);
  }

  function renderSubsidebar() {
    if (!state.view) {
      el.subsidebar.hidden = true;
      el.subsidebar.innerHTML = "";
      return;
    }

    el.subsidebar.hidden = false;
    el.subsidebar.innerHTML = `<header class="sub-head">
        <h2>${VIEW_TITLES[state.view]}</h2>
        <button class="sub-icon-btn" id="sub-close" title="Close panel" aria-label="Close panel">${icon("close", 13)}</button>
      </header>
      <div class="sub-body">${subBody()}</div>`;

    $("#sub-close").onclick = () => {
      state.view = null;
      subQuery = "";
      subOpenForm = false;
      render();
    };

    const chatSearch = $("#sub-chat-search");
    if (chatSearch) {
      chatSearch.oninput = () => {
        subQuery = chatSearch.value;
        renderSubsidebar();
        refocus("#sub-chat-search");
      };
    }

    const risetSearch = $("#sub-riset-search");
    if (risetSearch) {
      risetSearch.oninput = () => {
        subQuery = risetSearch.value;
        renderSubsidebar();
        refocus("#sub-riset-search");
      };
    }

    el.subsidebar.querySelectorAll("[data-session]").forEach((button) => {
      button.onclick = () => openSession(button.dataset.session);
    });

    el.subsidebar.querySelectorAll("[data-restore]").forEach((button) => {
      button.onclick = () => {
        const session = state.sessions.find((item) => item.id === button.dataset.restore);
        if (!session) return;
        session.archived = false;
        session.ts = Date.now();
        render();
      };
    });

    const newPrompt = $("#sub-new-prompt");
    if (newPrompt) {
      newPrompt.onclick = () => {
        subOpenForm = !subOpenForm;
        renderSubsidebar();
        if (subOpenForm) refocus("#prompt-title");
      };
    }

    const promptBody = $("#prompt-body");
    if (promptBody) {
      promptBody.oninput = () => {
        const saveButton = $("#prompt-save");
        if (saveButton) saveButton.disabled = !promptBody.value.trim();
      };
    }

    const promptCancel = $("#prompt-cancel");
    if (promptCancel) {
      promptCancel.onclick = () => {
        subOpenForm = false;
        renderSubsidebar();
      };
    }

    const promptSave = $("#prompt-save");
    if (promptSave) {
      promptSave.onclick = () => {
        const title = ($("#prompt-title").value || "").trim();
        const body = ($("#prompt-body").value || "").trim();
        if (!body) return;
        state.prompts.unshift({
          id: `p${Date.now()}`,
          title: title || "Untitled",
          body,
        });
        subOpenForm = false;
        render();
      };
    }

    el.subsidebar.querySelectorAll("[data-prompt-use]").forEach((button) => {
      button.onclick = () => {
        const prompt = state.prompts.find((item) => item.id === button.dataset.promptUse);
        if (!prompt) return;
        el.input.value = prompt.body;
        el.input.focus();
        resizeInput();
        syncSend();
      };
    });

    el.subsidebar.querySelectorAll("[data-prompt-copy]").forEach((button) => {
      button.onclick = async () => {
        const prompt = state.prompts.find((item) => item.id === button.dataset.promptCopy);
        if (!prompt) return;
        try {
          await navigator.clipboard.writeText(prompt.body);
          button.innerHTML = icon("check", 11);
          setTimeout(() => (button.innerHTML = icon("copy", 11)), 1200);
        } catch (_) {
          /* clipboard unavailable */
        }
      };
    });

    el.subsidebar.querySelectorAll("[data-prompt-del]").forEach((button) => {
      button.onclick = () => {
        state.prompts = state.prompts.filter((item) => item.id !== button.dataset.promptDel);
        render();
      };
    });
  }

  /* ── Sending and streaming ─────────────────────────────────────────────── */
  function resizeInput() {
    el.input.style.height = "auto";
    el.input.style.height = Math.min(160, el.input.scrollHeight) + "px";
  }

  function newSession() {
    const session = {
      id: `s${Date.now()}`,
      title: "New chat",
      preview: "No messages yet",
      pinned: false,
      archived: false,
      ts: Date.now(),
      messages: [],
    };
    state.sessions.unshift(session);
    state.activeId = session.id;
    pills = samplePills();
    welcomePart = dayPart();
    welcomeCopy = sampleWelcome(welcomePart);
    return session;
  }

  function ensureSession() {
    const session = active();
    if (session && !session.archived && !session.messages.length) return session;
    return newSession();
  }

  function openSession(id) {
    if (streaming) return;
    const session = state.sessions.find((item) => item.id === id);
    if (!session) return;
    state.activeId = session.id;
    render();
    el.input.focus();
  }

  function updateThoughtClock() {
    if (!thinkingMessage || thinkingMessage.thoughtMs !== undefined) return;
    if (thinkingMessage.thoughtStartedAt === undefined) return;
    const label = document.querySelector(".msg.assistant:last-child .think-label");
    if (!label) return;
    label.textContent = `Thinking for ${formatThoughtClock(
      Date.now() - thinkingMessage.thoughtStartedAt,
    )}`;
  }

  function startStream(session, message, prompt) {
    const full = reply(prompt);
    const thought = thoughtFor(prompt);
    let index = 0;

    const beginAnswer = () => {
      streaming = window.setInterval(() => {
        index = Math.min(full.length, index + 2 + Math.floor(Math.random() * 5));
        message.content = full.slice(0, index);

        const node = document.querySelector(
          `.msg[data-msg="${session.messages.length - 1}"] [data-md]`,
        );
        if (node) {
          node.innerHTML = mdToHtml(message.content) + '<span class="caret"></span>';
          const stream = $("#stream");
          if (stream) stream.scrollTop = stream.scrollHeight;
        }

        if (index >= full.length) stopStreaming(session, message);
      }, 24);

      syncSend();
      renderStatus();
    };

    if (!thought) {
      beginAnswer();
      return;
    }

    message.thinking = thought;
    message.thoughtStartedAt = Date.now();
    thinkingMessage = message;
    openThoughts.add(`${session.id}:${session.messages.length - 1}`);
    renderView();
    const liveStream = $("#stream");
    if (liveStream) liveStream.scrollTop = liveStream.scrollHeight;
    thoughtTimer = window.setInterval(updateThoughtClock, 1000);
    streaming = thoughtTimer;
    syncSend();
    renderStatus();

    thoughtTimeout = window.setTimeout(() => {
      thoughtTimeout = null;
      if (!message.streaming) return;
      message.thoughtMs = Date.now() - message.thoughtStartedAt;
      thinkingMessage = null;
      if (thoughtTimer) {
        window.clearInterval(thoughtTimer);
        thoughtTimer = null;
      }
      renderView();
      const settled = $("#stream");
      if (settled) settled.scrollTop = settled.scrollHeight;
      beginAnswer();
    }, 2600);
  }

  function stopStreaming(session, message) {
    if (streaming) window.clearInterval(streaming);
    streaming = null;
    if (thoughtTimer) {
      window.clearInterval(thoughtTimer);
      thoughtTimer = null;
    }
    if (thoughtTimeout) {
      window.clearTimeout(thoughtTimeout);
      thoughtTimeout = null;
    }
    if (
      message.thinking &&
      message.thoughtMs === undefined &&
      message.thoughtStartedAt !== undefined
    ) {
      message.thoughtMs = Date.now() - message.thoughtStartedAt;
    }
    thinkingMessage = null;
    message.streaming = false;
    if (!message.content) message.content = "Stopped.";
    session.ts = Date.now();
    session.preview = previewOf(session);
    render();
  }

  function cancelStream() {
    const session = active();
    const message = session && session.messages[session.messages.length - 1];
    if (streaming && message && message.streaming) stopStreaming(session, message);
  }

  function send() {
    if (streaming || state.boot !== "ready") return;
    const text = el.input.value.trim();
    if (!text) return;

    const session = ensureSession();
    if (!session.messages.length) {
      session.title = text.length > 46 ? text.slice(0, 46) + "…" : text;
    }

    session.messages.push({ role: "user", content: text });
    session.ts = Date.now();
    session.preview = text.slice(0, 90);
    el.input.value = "";
    resizeInput();
    render();

    const message = { role: "assistant", content: "", streaming: true };
    session.messages.push(message);
    render();

    startStream(session, message, text);
    el.input.focus();
  }

  function regenerate(index) {
    if (streaming) return;
    const session = active();
    if (!session) return;
    const message = session.messages[index];
    if (!message || message.role !== "assistant") return;

    const user = session.messages
      .slice(0, index)
      .reverse()
      .find((item) => item.role === "user");
    if (!user) return;

    session.messages = session.messages.slice(0, index);
    const next = { role: "assistant", content: "", streaming: true };
    session.messages.push(next);
    render();
    startStream(session, next, user.content);
  }

  /* ── Popovers ──────────────────────────────────────────────────────────── */
  function closePopover() {
    el.popover.hidden = true;
    el.popover.innerHTML = "";
    if (popoverAnchor) {
      popoverAnchor.setAttribute("aria-expanded", "false");
      popoverAnchor.classList.remove("is-open");
    }
    popoverAnchor = null;
  }

  function openPopover(anchor, html, placement) {
    if (popoverAnchor === anchor && !el.popover.hidden) {
      closePopover();
      return;
    }
    closePopover();
    popoverAnchor = anchor;
    anchor.setAttribute("aria-expanded", "true");
    if (anchor.classList.contains("bar-btn")) anchor.classList.add("is-open");
    el.popover.innerHTML = html;
    el.popover.hidden = false;

    const rect = anchor.getBoundingClientRect();
    const width = el.popover.offsetWidth;
    const height = el.popover.offsetHeight;
    let left = rect.left;
    if (left + width > window.innerWidth - 8) left = window.innerWidth - width - 8;
    let top = placement === "down" ? rect.bottom + 6 : rect.top - height - 6;
    if (top < 8) top = rect.bottom + 6;
    if (top + height > window.innerHeight - 8) top = rect.top - height - 6;
    el.popover.style.left = Math.max(8, left) + "px";
    el.popover.style.top = Math.max(8, top) + "px";

    el.popover.querySelectorAll("[data-workspace]").forEach((button) => {
      button.onclick = () => {
        const view = button.dataset.workspace;
        state.view = state.view === view ? null : view;
        subQuery = "";
        subOpenForm = false;
        closePopover();
        render();
      };
    });

    el.popover.querySelectorAll("[data-insert]").forEach((button) => {
      button.onclick = () => {
        const mode = button.dataset.insert;
        closePopover();
        if (mode === "paste") void pasteFromClipboard();
        else insertCodeBlock();
      };
    });

    el.popover.querySelectorAll("[data-saved-prompt]").forEach((button) => {
      button.onclick = () => {
        const prompt = (state.prompts || [])[Number(button.dataset.savedPrompt)];
        closePopover();
        if (prompt) applySavedPrompt(prompt);
      };
    });

    el.popover.querySelectorAll("[data-tier]").forEach((button) => {
      button.onclick = () => {
        state.tier = button.dataset.tier;
        closePopover();
        render();
      };
    });

    el.popover.querySelectorAll("[data-mode]").forEach((button) => {
      button.onclick = () => {
        state.mode = button.dataset.mode;
        closePopover();
        render();
      };
    });

    el.popover.querySelectorAll("[data-manage]").forEach((button) => {
      button.onclick = () => {
        closePopover();
        tab = "Model";
        el.settings.hidden = false;
        renderSettings();
      };
    });
  }

  function workspacesHtml() {
    return WORKSPACES.map(
      (item) => `<button class="popover-item with-icon ${state.view === item[0] ? "active" : ""}" role="menuitem" data-workspace="${item[0]}">
        ${drawIcon(item[1], 18)}<span class="popover-item-text">${item[2]}</span>
        ${state.view === item[0] ? icon("check", 13) : ""}
      </button>`,
    ).join("");
  }

  function attachHtml() {
    const prompts = (state.prompts || []).slice(0, 3);
    return `<div class="popover-label">Attach</div>
      <button class="popover-item" disabled><span class="popover-item-main">${icon("file", 14)} Files…</span><span class="popover-hint">desktop</span></button>
      <button class="popover-item" disabled><span class="popover-item-main">${icon("folder", 14)} Folder…</span><span class="popover-hint">desktop</span></button>
      <button class="popover-item" disabled title="Image input isn't available yet"><span class="popover-item-main">${icon("image", 14)} Images…</span><span class="popover-hint">needs vision</span></button>
      <div class="popover-sep"></div>
      <div class="popover-label">Insert</div>
      <button class="popover-item" data-insert="paste"><span class="popover-item-main">${icon("copy", 14)} Paste text</span><span class="popover-hint">clipboard</span></button>
      <button class="popover-item" data-insert="code"><span class="popover-item-main">${icon("code", 14)} Code block</span><span class="popover-hint">fenced</span></button>
      ${
        prompts.length
          ? `<div class="popover-sep"></div><div class="popover-label">Saved prompts</div>` +
            prompts
              .map(
                (prompt, index) =>
                  `<button class="popover-item" data-saved-prompt="${index}" title="${escAttr(prompt.body)}">
                    <span class="popover-item-main">${icon("book", 14)}<span class="popover-file">${esc(prompt.title)}</span></span>
                  </button>`,
              )
              .join("")
          : ""
      }`;
  }

  function insertIntoComposer(text, selectStart, selectEnd) {
    const start = el.input.selectionStart ?? el.input.value.length;
    const end = el.input.selectionEnd ?? start;
    el.input.value = el.input.value.slice(0, start) + text + el.input.value.slice(end);
    el.input.focus();
    const from = start + (selectStart ?? text.length);
    const to = start + (selectEnd ?? text.length);
    el.input.setSelectionRange(from, to);
    resizeInput();
    syncSend();
  }

  async function pasteFromClipboard() {
    try {
      const text = await navigator.clipboard.readText();
      if (text) insertIntoComposer(text);
      else el.input.focus();
    } catch (_) {
      el.input.focus();
    }
  }

  function insertCodeBlock() {
    insertIntoComposer("```\n\n```", 4, 4);
  }

  function applySavedPrompt(prompt) {
    const draft = el.input.value;
    el.input.value = draft.trim()
      ? `${draft.replace(/\s+$/, "")}\n\n${prompt.body}`
      : prompt.body;
    el.input.focus();
    el.input.setSelectionRange(el.input.value.length, el.input.value.length);
    resizeInput();
    syncSend();
  }

  function modelHtml() {
    return `<div class="popover-label">Model</div>
      <button class="model-row active" role="menuitem">
        <img src="assets/demo/Dizel.png" alt="">
        <span class="model-row-text"><b>${MODEL.name}</b><small>${MODEL.params} · local</small></span>
        ${icon("check", 13)}
      </button>
      <div class="popover-label">Preview</div>
      <button class="model-row" role="menuitem" disabled>
        <img src="assets/demo/Dizel.png" alt="">
        <span class="model-row-text"><b>Dizel 1.6 Flash Preview</b><small>weights unpublished</small></span>
      </button>
      <div class="popover-sep"></div>
      <button class="popover-item" role="menuitem" data-manage><span class="popover-item-main">Manage models…</span></button>`;
  }

  function tierHtml() {
    return `<div class="popover-label">Tier</div>
      ${TIERS.map(
        (name) => `<button class="popover-item ${state.tier === name ? "active" : ""}" role="menuitem" data-tier="${name}">
          <span class="popover-item-main">${name}</span>
          ${state.tier === name ? icon("check", 13) : ""}
        </button>`,
      ).join("")}
      <div class="popover-sep"></div>
      <div class="popover-label">Mode</div>
      ${MODES.map(
        (name) => `<button class="popover-item ${state.mode === name ? "active" : ""}" role="menuitem" data-mode="${name}">
          <span class="popover-item-main">${name}</span>
          ${state.mode === name ? icon("check", 13) : ""}
        </button>`,
      ).join("")}`;
  }

  /* ── Settings ──────────────────────────────────────────────────────────── */
  const TABS = ["Chat", "Model", "Appearance", "Speech", "Shortcuts", "About"];
  let tab = "Chat";

  const HINTS = {
    Chat: "How the assistant is framed, and how Enter behaves.",
    Model: "Tier and response mode. The desktop app also picks a checkpoint here.",
    Appearance: "Three themes, ported from the desktop tokens.",
    Speech: "Whisper model, language, task and vocabulary. Stored in this browser.",
    Shortcuts: "Global shortcuts, as listed in the desktop app.",
    About: "Version and licensing.",
  };

  function segmentedHtml(id, items, current, dataKey, label) {
    return `<div class="segmented" id="${id}" role="radiogroup" aria-label="${escAttr(label)}">${items
      .map(([value, text]) => {
        const checked = value === current;
        return `<button type="button" class="segmented-item" role="radio" data-${dataKey}="${escAttr(
          value,
        )}" aria-checked="${checked}" tabindex="${checked ? "0" : "-1"}">${esc(text)}</button>`;
      })
      .join("")}</div>`;
  }

  function bindSegmented(id, dataKey, onPick) {
    const group = $("#" + id);
    if (!group) return;
    const radios = Array.from(group.querySelectorAll('[role="radio"]'));

    const choose = (button, moveFocus) => {
      radios.forEach((radio) => {
        const active = radio === button;
        radio.setAttribute("aria-checked", String(active));
        radio.tabIndex = active ? 0 : -1;
      });
      if (moveFocus) button.focus();
      onPick(button.dataset[dataKey]);
    };

    radios.forEach((button) => {
      button.onclick = () => choose(button, false);
      button.onkeydown = (event) => {
        const index = radios.indexOf(button);
        let target = null;
        if (event.key === "ArrowRight" || event.key === "ArrowDown") {
          target = radios[(index + 1) % radios.length];
        } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
          target = radios[(index - 1 + radios.length) % radios.length];
        } else if (event.key === "Home") {
          target = radios[0];
        } else if (event.key === "End") {
          target = radios[radios.length - 1];
        }
        if (!target) return;
        event.preventDefault();
        choose(target, true);
      };
    });
  }

  function renderSettings() {
    el.settingsTabs.innerHTML = TABS.map(
      (name) =>
        `<button class="modal-tab ${name === tab ? "active" : ""}" data-tab="${name}">${name}</button>`,
    ).join("");
    el.settingsTabs.querySelectorAll("[data-tab]").forEach((button) => {
      button.onclick = () => {
        tab = button.dataset.tab;
        renderSettings();
      };
    });

    let body = `<h3>${tab}</h3><p class="hint">${HINTS[tab]}</p>`;

    if (tab === "Chat") {
      body += `<div class="section-title">Checkpoint</div>
        <div class="field">
          <label for="ckpt-chat">Model file</label>
          <input id="ckpt-chat" type="text" value="— none selected — (browser demo)" disabled>
        </div>
        <div class="field">
          <label for="dev">Device</label>
          <input id="dev" type="text" value="DEMO" disabled>
        </div>

        <div class="section-title">Provider</div>
        <div class="prov-grid">
          ${PROVIDERS.map(
            ([slug, label, file]) =>
              `<button class="prov ${state.provider === slug ? "active" : ""}" data-prov="${slug}" title="${label}">
              <img src="assets/${slug === "local" ? "demo" : "providers"}/${file}" alt="">${label}
            </button>`,
          ).join("")}
        </div>
        <p class="hint" style="margin-bottom:18px">The desktop app stores keys encrypted next to your settings and routes requests through the chosen provider. This page calls nothing.</p>

        <div class="section-title">Conversation</div>
        <div class="field">
          <label for="sp">System prompt</label>
          <textarea id="sp" spellcheck="false">${esc(state.systemPrompt)}</textarea>
        </div>
        <div class="switch-row">
          <span>Enter sends the message
            <small>Off: Ctrl+Enter sends, Enter inserts a new line.</small>
          </span>
          <button class="switch" id="sw-enter" role="switch" aria-checked="${state.enterToSend}" aria-label="Enter sends"></button>
        </div>`;
    }

    if (tab === "Model") {
      body += `<div class="field">
          <label>Tier</label>
          ${segmentedHtml("seg-tier", TIERS.map((name) => [name, name]), state.tier, "tier", "Model tier")}
        </div>
        <div class="field">
          <label>Mode</label>
          ${segmentedHtml("seg-mode", MODES.map((name) => [name, name]), state.mode, "mode", "Response mode")}
        </div>
        <div class="field">
          <label for="ckpt">Checkpoint</label>
          <input id="ckpt" type="text" value="— none selected — (browser demo)" disabled>
        </div>
        <div class="field">
          <label for="ctx">Context length</label>
          <input id="ctx" type="text" value="2048" disabled>
        </div>`;
    }

    if (tab === "Appearance") {
      body += `<div class="field">
          <label>Theme</label>
          ${segmentedHtml("seg-theme", THEMES, state.theme, "theme", "Theme")}
        </div>
        <p class="hint">Themes are the desktop app's three token sets: dark sampled from the running app, light and dark-blue from <code>theme/colors.py</code>.</p>`;
    }

    if (tab === "Speech") {
      body += `<div class="section-title">Nova dictation</div>
        <p class="hint" style="margin-bottom:14px">Speech-to-text runs in the desktop app's sidecar. These choices are demo preferences, kept in this browser.</p>
        <div class="field">
          <label for="whisper-model">Whisper model</label>
          <select id="whisper-model">
            ${SPEECH_MODELS.map(
              ([value, label]) =>
                `<option value="${value}"${state.speech.model === value ? " selected" : ""}>${label}</option>`,
            ).join("")}
          </select>
          <small class="field-note">Larger is more accurate but loads slower and needs more RAM.</small>
        </div>
        <div class="field">
          <label for="whisper-language">Language</label>
          <select id="whisper-language">
            ${SPEECH_LANGUAGES.map(
              ([value, label]) =>
                `<option value="${value}"${state.speech.language === value ? " selected" : ""}>${label}</option>`,
            ).join("")}
          </select>
          <small class="field-note">Auto lets Whisper detect the spoken language.</small>
        </div>
        <div class="field">
          <label for="whisper-task">Task</label>
          <select id="whisper-task">
            ${SPEECH_TASKS.map(
              ([value, label]) =>
                `<option value="${value}"${state.speech.task === value ? " selected" : ""}>${label}</option>`,
            ).join("")}
          </select>
          <small class="field-note">Transcribe keeps the source language; Translate returns English.</small>
        </div>
        <div class="section-title">Vocabulary</div>
        <div class="field">
          <label for="whisper-prompt">Initial prompt</label>
          <textarea id="whisper-prompt" rows="2" maxlength="400" spellcheck="false" placeholder="Dizel, GGUF, safetensors, Dan…">${esc(
            state.speech.prompt,
          )}</textarea>
          <small class="field-note">Names, acronyms and jargon to bias the next dictation. Up to 400 characters.</small>
        </div>`;
    }

    if (tab === "Shortcuts") {
      body += `<div>
        ${[
          ["New chat", "Ctrl N"],
          ["Focus the composer", "Ctrl K"],
          ["Collapse the sidebar", "Ctrl B"],
          ["Open settings", "Ctrl ,"],
          ["Send", "Enter"],
          ["New line", "Shift Enter"],
          ["Close a menu", "Esc"],
        ]
          .map(([label, keys]) => `<div class="shortcut"><span>${label}</span><kbd>${keys}</kbd></div>`)
          .join("")}
      </div>`;
    }

    if (tab === "About") {
      body += `<dl class="kv">
        <dt>Version</dt><dd>2.0.0 (desktop shell)</dd>
        <dt>Shell</dt><dd>Electron 33 · React 18 · Vite 5</dd>
        <dt>Engine</dt><dd>Python sidecar · ChatManager</dd>
        <dt>Models</dt><dd>Dizel 1.5 Distill · Dizel 1.6 Flash Preview (unpublished)</dd>
        <dt>Licence</dt><dd>MIT</dd>
        <dt>Author</dt><dd>D4niel-dev</dd>
      </dl>
      <p class="hint" style="margin-top:16px">This page is a browser replica. It runs no model, keeps your chats in localStorage, and is not the shipping app.</p>`;
    }

    el.settingsPanel.innerHTML = body;

    const swEnter = $("#sw-enter");
    if (swEnter) {
      swEnter.onclick = () => {
        state.enterToSend = !state.enterToSend;
        swEnter.setAttribute("aria-checked", String(state.enterToSend));
        save();
      };
    }

    const sp = $("#sp");
    if (sp) {
      sp.onchange = () => {
        state.systemPrompt = sp.value;
        save();
      };
    }

    const whisperModel = $("#whisper-model");
    if (whisperModel) {
      whisperModel.onchange = () => {
        state.speech.model = whisperModel.value;
        save();
      };
    }
    const whisperLanguage = $("#whisper-language");
    if (whisperLanguage) {
      whisperLanguage.onchange = () => {
        state.speech.language = whisperLanguage.value;
        save();
      };
    }
    const whisperTask = $("#whisper-task");
    if (whisperTask) {
      whisperTask.onchange = () => {
        state.speech.task = whisperTask.value;
        save();
      };
    }
    const whisperPrompt = $("#whisper-prompt");
    if (whisperPrompt) {
      whisperPrompt.oninput = () => {
        state.speech.prompt = whisperPrompt.value.slice(0, 400);
        save();
      };
    }

    el.settingsPanel.querySelectorAll("[data-prov]").forEach((button) => {
      button.onclick = () => {
        state.provider = button.dataset.prov;
        renderSettings();
        save();
      };
    });

    bindSegmented("seg-theme", "theme", (value) => {
      state.theme = value;
      render();
    });

    bindSegmented("seg-tier", "tier", (value) => {
      state.tier = value;
      render();
    });

    bindSegmented("seg-mode", "mode", (value) => {
      state.mode = value;
      render();
    });
  }

  /* ── Wiring ────────────────────────────────────────────────────────────── */
  el.input.addEventListener("input", () => {
    resizeInput();
    syncSend();
  });

  el.composer.addEventListener("submit", (event) => {
    event.preventDefault();
    if (streaming) {
      cancelStream();
      return;
    }
    send();
  });

  el.input.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || event.isComposing) return;
    const modifier = event.ctrlKey || event.metaKey;
    const shouldSend = state.enterToSend ? !event.shiftKey : modifier;
    if (shouldSend) {
      event.preventDefault();
      if (streaming) return;
      send();
    }
  });

  $("#new-chat").onclick = () => {
    newSession();
    render();
    el.input.focus();
  };

  $("#nav-chats").onclick = () => {
    state.view = state.view === "Chats" ? null : "Chats";
    subQuery = "";
    subOpenForm = false;
    render();
  };

  $("#nav-workspaces").onclick = (event) => {
    event.stopPropagation();
    openPopover(event.currentTarget, workspacesHtml(), "down");
  };

  $("#btn-plus").onclick = (event) => {
    event.stopPropagation();
    openPopover(event.currentTarget, attachHtml());
  };

  $("#btn-model").onclick = (event) => {
    event.stopPropagation();
    openPopover(event.currentTarget, modelHtml());
  };

  $("#btn-tier").onclick = (event) => {
    event.stopPropagation();
    openPopover(event.currentTarget, tierHtml());
  };

  $("#btn-mic").onclick = () => {
    const previous = el.input.placeholder;
    el.input.focus();
    el.input.placeholder = "Dictation is only available in the desktop app…";
    setTimeout(() => {
      el.input.placeholder = previous;
    }, 1800);
  };

  el.list.addEventListener("click", (event) => {
    const row = event.target.closest(".sb-row");
    if (!row) return;
    const id = row.dataset.id;
    const session = state.sessions.find((item) => item.id === id);
    if (!session) return;
    const action = event.target.closest("[data-act]")?.dataset.act;

    if (action === "pin") {
      session.pinned = !session.pinned;
      render();
      return;
    }

    if (action === "archive") {
      session.archived = true;
      if (state.activeId === id) {
        const next = state.sessions.find((item) => !item.archived);
        state.activeId = next ? next.id : null;
      }
      render();
      return;
    }

    if (action === "delete") {
      if (confirmId !== id) {
        confirmId = id;
        window.clearTimeout(confirmTimer);
        confirmTimer = window.setTimeout(() => {
          confirmId = null;
          renderSidebar();
        }, 2600);
        renderSidebar();
        return;
      }
      window.clearTimeout(confirmTimer);
      confirmId = null;
      state.sessions = state.sessions.filter((item) => item.id !== id);
      if (state.activeId === id) {
        const next = state.sessions.find((item) => !item.archived);
        state.activeId = next ? next.id : null;
      }
      render();
      return;
    }

    openSession(id);
  });

  $("#toggle-recent").onclick = () => {
    state.showRecent = !state.showRecent;
    render();
  };

  el.search.addEventListener("input", renderSidebar);

  $("#sb-collapse").onclick = () => {
    state.collapsed = !state.collapsed;
    render();
  };

  $("#sb-foot").onclick = () => {
    $("#profile-name").value = state.username;
    el.profile.hidden = false;
    $("#profile-name").focus();
  };

  $("#profile-close").onclick = () => (el.profile.hidden = true);
  $("#profile-save").onclick = () => {
    const name = $("#profile-name").value.trim();
    if (name) state.username = name;
    el.profile.hidden = true;
    render();
  };
  el.profile.addEventListener("click", (event) => {
    if (event.target === el.profile) el.profile.hidden = true;
  });

  $("#open-settings").onclick = () => {
    tab = "Chat";
    el.settings.hidden = false;
    renderSettings();
  };
  $("#settings-close").onclick = () => (el.settings.hidden = true);
  el.settings.addEventListener("click", (event) => {
    if (event.target === el.settings) el.settings.hidden = true;
  });

  $("#toggle-statusbar").onclick = () => {
    state.showStatusbar = !state.showStatusbar;
    render();
  };

  $("#demo-bar-x").onclick = () => {
    $("#demo-bar").remove();
    sessionStorage.setItem("dizel-demo-notice", "hidden");
  };
  if (sessionStorage.getItem("dizel-demo-notice") === "hidden") $("#demo-bar").remove();

  document.addEventListener("click", (event) => {
    if (!el.popover.hidden && !el.popover.contains(event.target) && event.target !== popoverAnchor) {
      closePopover();
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      if (!el.popover.hidden) closePopover();
      if (!el.settings.hidden) el.settings.hidden = true;
      if (!el.profile.hidden) el.profile.hidden = true;
      return;
    }
    if (!(event.ctrlKey || event.metaKey)) return;
    const key = event.key.toLowerCase();
    if (key === "n") {
      event.preventDefault();
      $("#new-chat").click();
    } else if (key === "k") {
      event.preventDefault();
      el.input.focus();
    } else if (key === "b") {
      event.preventDefault();
      $("#sb-collapse").click();
    } else if (event.key === ",") {
      event.preventDefault();
      $("#open-settings").click();
    }
  });

  /* ── Boot ──────────────────────────────────────────────────────────────── */
  // A phone gets the collapsed rail by default; the toggle still opens the panel.
  const narrow = window.matchMedia("(max-width: 900px)");

  const collapseForNarrow = (matches) => {
    if (!matches) return;
    state.collapsed = true;
    state.showRecent = false;
    render();
  };

  if (narrow.matches) {
    state.collapsed = true;
    state.showRecent = false;
  }

  if (narrow.addEventListener) narrow.addEventListener("change", (event) => collapseForNarrow(event.matches));

  render();
  resizeInput();

  // Crossing noon or 6pm re-rolls the headline into the new time bucket.
  window.setInterval(() => {
    const nextPart = dayPart();
    if (nextPart === welcomePart) return;
    welcomePart = nextPart;
    welcomeCopy = sampleWelcome(nextPart);
    const session = active();
    if (!session || !session.messages.length) render();
  }, 60000);

  window.setInterval(() => {
    const mins = Math.floor((Date.now() - BOOTED_AT) / 60000);
    const node = $("#sb-time");
    if (node) node.textContent = `${Math.floor(mins / 60)}h ${mins % 60}m`;
  }, 30000);

  const bootStart = Date.now();
  loaderTimer = window.setInterval(() => {
    const node = $("#loader-elapsed");
    if (node) node.textContent = `${Math.floor((Date.now() - bootStart) / 1000)}s`;
  }, 1000);

  setTimeout(() => {
    state.boot = "loading";
    render();
  }, 550);

  setTimeout(() => {
    state.boot = "ready";
    if (loaderTimer) window.clearInterval(loaderTimer);
    render();
    resizeInput();
  }, 1600);
})();
