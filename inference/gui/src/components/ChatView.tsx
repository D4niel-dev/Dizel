import { useEffect, useLayoutEffect, useRef, useState } from "react";

import { Composer } from "./Composer";
import { Icon } from "./Icon";
import { MessageBubble } from "./MessageBubble";
import { ModelLoader } from "./ModelLoader";
import { dayPart, greeting } from "../services/greeting";
import { fillWelcome, sampleWelcome } from "../services/welcome";
import type { IconName } from "./Icon";
import { useChat } from "../store/chatStore";

/**
 * Starters for a text-only model — Dizel 1.6 Flash Preview has no vision
 * encoder, so "Create Image" would promise something it cannot do.
 */
type Suggestion = {
  label: string;
  icon: IconName;
  prompt: string;
};

/** Three at a time, drawn from the pool — the row is a prompt, not a menu. */
const PILLS_SHOWN = 3;

function samplePills(pool: Suggestion[], count: number): Suggestion[] {
  const copy = [...pool];
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy.slice(0, Math.min(count, copy.length));
}

const SUGGESTIONS: Suggestion[] = [
  { label: "Write Code", icon: "code", prompt: "Write a function that " },
  { label: "Explain a concept", icon: "book", prompt: "Explain how " },
  { label: "Make a plan", icon: "sparkles", prompt: "Make a plan for " },
  { label: "Summarise", icon: "archive", prompt: "Summarise this: " },
  { label: "Translate", icon: "globe", prompt: "Translate to English: " },
  { label: "Fix an error", icon: "alert", prompt: "Why does this fail: " },
];

export function ChatView() {
  const messages = useChat((s) => s.messages);
  const username = useChat((s) => s.username);
  const greetingStyle = useChat((s) => s.profile.greetingStyle);
  const booted = useChat((s) => s.booted);
  const loadStatus = useChat((s) => s.status?.load?.status);
  const recent = useChat((s) => s.recent);
  const openSession = useChat((s) => s.openSession);
  const setDraft = useChat((s) => s.setDraft);
  const chatEpoch = useChat((s) => s.chatEpoch);
  const [pills, setPills] = useState<Suggestion[]>(() => samplePills(SUGGESTIONS, PILLS_SHOWN));
  const [part, setPart] = useState(() => dayPart());
  const [welcomeCopy, setWelcomeCopy] = useState(() => sampleWelcome(part));

  // A fresh chat gets a fresh draw. Re-rolling on every render would make the
  // row flicker; `chatEpoch` increments once per New Chat, which is exactly the
  // moment the row should change — even if the chat was already empty.
  // Crossing noon or 6pm re-rolls only the headline's time bucket.
  useEffect(() => {
    setPills(samplePills(SUGGESTIONS, PILLS_SHOWN));
    setWelcomeCopy(sampleWelcome(part));
  }, [chatEpoch, part]);

  useEffect(() => {
    const id = window.setInterval(() => {
      setPart((previous) => {
        const next = dayPart();
        return next === previous ? previous : next;
      });
    }, 60_000);
    return () => window.clearInterval(id);
  }, []);

  const scrollRef = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);

  const [findOpen, setFindOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [cursor, setCursor] = useState(0);
  const [flashId, setFlashId] = useState<string | null>(null);

  // Match at message granularity. Highlighting inside rendered Markdown would
  // mean re-parsing the tree; jumping to the message that contains the term is
  // what people actually use this for.
  const needle = query.trim().toLowerCase();
  const hits = needle
    ? messages
        .map((message, index) => ({ message, index }))
        .filter((entry) => entry.message.content.toLowerCase().includes(needle))
    : [];

  const jumpTo = (nextIndex: number) => {
    if (hits.length === 0) return;
    const wrapped = ((nextIndex % hits.length) + hits.length) % hits.length;
    setCursor(wrapped);
    const id = hits[wrapped].message.id;
    const node = document.querySelector(`[data-mid="${id}"]`);
    if (node) {
      node.scrollIntoView({ block: "center", behavior: "smooth" });
      setFlashId(id);
      window.setTimeout(() => setFlashId(null), 1200);
    }
  };

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const mod = event.ctrlKey || event.metaKey;
      if (mod && event.key.toLowerCase() === "f") {
        event.preventDefault();
        setFindOpen(true);
        return;
      }
      if (event.key === "Escape" && findOpen) {
        setFindOpen(false);
        setQuery("");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [findOpen]);

  useEffect(() => {
    if (hits.length > 0) jumpTo(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [needle]);
  const [awayFromBottom, setAwayFromBottom] = useState(false);

  // `follow` is the user's choice; `stickToBottom` is whether they are currently
  // near the end. Both have to agree before the view jumps.
  const follow = useChat((s) => s.prefs.autoScroll);
  const showSuggestions = useChat((s) => s.prefs.showSuggestions);

  useEffect(() => {
    const el = scrollRef.current;
    if (el && follow && stickToBottom.current) el.scrollTop = el.scrollHeight;
  }, [messages, follow]);

  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 90;
    stickToBottom.current = atBottom;
    setAwayFromBottom(!atBottom);
  };

  const jumpToLatest = () => {
    const el = scrollRef.current;
    if (!el) return;
    stickToBottom.current = true;
    setAwayFromBottom(false);
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  };

  const prefill = (prompt: string) => {
    setDraft(prompt);
    document.querySelector<HTMLTextAreaElement>(".composer textarea")?.focus();
  };

  // ── Composer hand-off ────────────────────────────────────────────────
  // The composer lives in a different container in each state (centred in the
  // welcome, docked at the bottom in a chat), so React unmounts one and mounts
  // the other and it used to teleport. FLIP: measure where it was, where it
  // landed, and animate the difference away.
  const isEmpty = messages.length === 0;
  const lastComposerRect = useRef<DOMRect | null>(null);
  const lastEmpty = useRef<boolean | null>(null);

  useLayoutEffect(() => {
    const el = document.querySelector<HTMLElement>(".composer");
    if (!el) return;

    const next = el.getBoundingClientRect();
    const previous = lastComposerRect.current;
    lastComposerRect.current = next;

    // Only when the state actually flipped. Running on every render meant a
    // stale rect could animate an element that had not moved.
    const flipped = lastEmpty.current !== null && lastEmpty.current !== isEmpty;
    lastEmpty.current = isEmpty;
    if (!flipped || !previous) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    const dx = previous.left - next.left;
    const dy = previous.top - next.top;
    const scale = previous.width / Math.max(1, next.width);
    if (Math.abs(dy) < 2 && Math.abs(dx) < 2 && Math.abs(scale - 1) < 0.01) return;

    // Put it back where it came from *before* the browser paints, then release.
    // Animating with the Web Animations API instead let one frame paint at the
    // destination first — the composer appeared to jump to the corner and then
    // slide into place.
    el.style.transformOrigin = "top left";
    el.style.transform = `translate(${dx}px, ${dy}px) scale(${scale})`;
    el.style.opacity = "0.55";

    requestAnimationFrame(() => {
      el.style.transition =
        "transform 420ms cubic-bezier(0.22, 1, 0.36, 1), opacity 200ms ease-out";
      el.style.transform = "none";
      el.style.opacity = "1";

      const settle = () => {
        el.style.transition = "";
        el.style.transform = "";
        el.style.opacity = "";
        el.style.transformOrigin = "";
        el.removeEventListener("transitionend", settle);
      };
      el.addEventListener("transitionend", settle);
    });
  }, [isEmpty]);

  if (isEmpty) {
    // Loading and ready share one structure — only the block above the composer
    // differs. Anything else makes the composer jump when the model finishes.
    const loading = !booted || loadStatus === "loading";

    return (
      <div className={loading ? "welcome is-loading" : "welcome"}>
        {/* The hero is a fixed-height band so the composer sits in the same place
            no matter what appears below it — the recall list arrives after boot
            and used to shove the whole column up by 100px. */}
        <div className="welcome-hero">
          {loading ? (
            <ModelLoader />
          ) : (
            <>
              <h1>
                {greetingStyle === "simple"
                  ? `Hey ${username}.`
                  : greetingStyle === "time"
                    ? `${greeting()}, ${username}!`
                    : fillWelcome(welcomeCopy.headline, {
                        greeting: greeting(),
                        name: username,
                      })}
              </h1>
              <p>
                {fillWelcome(welcomeCopy.subline, {
                  greeting: greeting(),
                  name: username,
                })}
              </p>
            </>
          )}

          <div className="welcome-composer">
            <Composer />
          </div>

          {showSuggestions && (
            <div className="action-row">
              {pills.map((item) => (
                <button
                  key={item.label}
                  className="action-pill"
                  onClick={() => prefill(item.prompt)}
                >
                  {item.label}
                  <Icon name={item.icon} size={14} />
                </button>
              ))}
            </div>
          )}
        </div>

        {recent.length > 0 && (
          <div className="recent-block">
            <div className="recent-head">Pick up where you left off</div>
            {recent.map((item) => (
              <button
                key={item.id}
                className="recent-row"
                onClick={() => void openSession(item.id)}
                title={item.summary}
              >
                <span className="recent-name">{item.title}</span>
                <span className="recent-sum">{item.summary}</span>
              </button>
            ))}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="stream-wrap">
      {findOpen && (
        <div className="find-bar" role="search">
          <Icon name="search" size={13} />
          <input
            autoFocus
            value={query}
            placeholder="Find in conversation"
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                jumpTo(event.shiftKey ? cursor - 1 : cursor + 1);
              }
              if (event.key === "Escape") {
                setFindOpen(false);
                setQuery("");
              }
            }}
          />
          <span className="find-count">
            {hits.length ? `${cursor + 1}/${hits.length}` : "0/0"}
          </span>
          <button className="find-btn" onClick={() => jumpTo(cursor - 1)} title="Previous (Shift+Enter)">
            <Icon name="chevronUp" size={12} />
          </button>
          <button className="find-btn" onClick={() => jumpTo(cursor + 1)} title="Next (Enter)">
            <Icon name="chevronDown" size={12} />
          </button>
          <button
            className="find-btn"
            onClick={() => {
              setFindOpen(false);
              setQuery("");
            }}
            title="Close (Esc)"
          >
            <Icon name="close" size={12} />
          </button>
        </div>
      )}
      <div className="stream" ref={scrollRef} onScroll={onScroll}>
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} flash={flashId === message.id} />
        ))}
      </div>

      {awayFromBottom && (
        <button className="jump-btn" onClick={jumpToLatest} title="Jump to the latest message">
          <Icon name="arrowUp" size={13} style={{ transform: "rotate(180deg)" }} />
          Latest
        </button>
      )}
    </div>
  );
}
