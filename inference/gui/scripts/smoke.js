/* Smoke test for the Dizel GUI.
 *
 * Runs inside the renderer after the app has booted. Returns JSON so the harness
 * logs it. Every check is something that has changed in the last few rounds, so a
 * failure here points at a specific feature rather than "the app is broken".
 */
new Promise((resolve) => {
  const store = window.__dizel;
  const results = [];
  const errors = [];

  // Catch anything thrown while we exercise the app. This cannot see errors from
  // before we were injected, but it does cover everything we touch below.
  window.addEventListener("error", (e) => errors.push(String(e.message)));
  window.addEventListener("unhandledrejection", (e) => errors.push(String(e.reason)));

  const check = (name, fn) => {
    try {
      const value = fn();
      results.push({ name, ok: value === true || (value && value.ok === true), detail: value });
    } catch (e) {
      results.push({ name, ok: false, detail: String(e && e.message) });
    }
  };

  const waitFor = (fn, tries = 60) =>
    new Promise((done) => {
      const tick = () => {
        let out = null;
        try {
          out = fn();
        } catch {
          out = null;
        }
        if (out || tries-- <= 0) return done(out);
        setTimeout(tick, 300);
      };
      tick();
    });

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  const boot = async () => {
    if (!store || typeof store.getState !== "function") {
      return resolve(JSON.stringify({ fatal: "no store on window.__dizel" }));
    }
    // Wait for the model load to settle; the composer is disabled until it does.
    await waitFor(() => {
      const load = store.getState().status && store.getState().status.load;
      return load && load.status && load.status !== "loading" ? true : null;
    }, 80);

    const api = store.getState();

    // ── 1. store surface ────────────────────────────────────────────────
    check("store: actions present", () => {
      const need = [
        "send", "regenerate", "beginEdit", "newChat", "showSibling",
        "showFork", "commitFork", "commitContinue", "continueLast",
        "openSession", "toggleStatusBar",
      ];
      const missing = need.filter((k) => typeof api[k] !== "function");
      return { ok: missing.length === 0, missing };
    });

    check("store: state fields", () => {
      const need = ["messages", "prefs", "chatEpoch", "pendingFork", "pendingContinue"];
      const missing = need.filter((k) => !(k in api));
      return { ok: missing.length === 0, missing, epoch: api.chatEpoch };
    });

    check("prefs: thinkingPause exists", () => {
      const v = api.prefs && api.prefs.thinkingPause;
      return { ok: ["off", "short", "long"].includes(v), value: v };
    });

    // ── 2. empty state ──────────────────────────────────────────────────
    api.newChat();
    await sleep(500);

    check("empty: three suggestion pills", () => {
      const pills = document.querySelectorAll(".action-pill");
      const labels = Array.from(pills).map((p) => p.textContent.trim());
      return { ok: pills.length === 3, count: pills.length, labels };
    });

    check("empty: export button hidden", () => {
      const btns = Array.from(document.querySelectorAll(".header-actions .hdr-btn"));
      const exportBtn = btns.find((b) => b.getAttribute("aria-label") === "Export conversation");
      return { ok: !exportBtn, found: Boolean(exportBtn) };
    });

    check("empty: no assistant bubbles", () => {
      const n = document.querySelectorAll(".msg.assistant").length;
      return { ok: n === 0, count: n };
    });

    // ── 3. pills re-roll per new chat ───────────────────────────────────
    const drawPills = () =>
      Array.from(document.querySelectorAll(".action-pill")).map((p) => p.textContent.trim()).join("|");

    const beforeEpoch = store.getState().chatEpoch;
    const firstDraw = drawPills();
    const draws = new Set([firstDraw]);
    for (let i = 0; i < 4; i += 1) {
      store.getState().newChat();
      await sleep(320);
      draws.add(drawPills());
    }

    check("pills: re-roll on new chat", () => {
      const afterEpoch = store.getState().chatEpoch;
      return {
        ok: afterEpoch > beforeEpoch && draws.size > 1,
        epochBefore: beforeEpoch,
        epochAfter: afterEpoch,
        distinctDraws: draws.size,
      };
    });

    // ── 4. popover placement follows state ──────────────────────────────
    const openVariant = async () => {
      const chips = document.querySelectorAll(".composer-bar .bar-chip");
      if (chips.length === 0) return { reason: "no chip in .composer-bar" };
      const chip = chips[chips.length - 1];
      chip.click();
      // Poll instead of sleeping a fixed 450ms. A cold first open can take longer
      // than that, which made this check fail on one run and pass on the next with
      // identical app code.
      const pop = await waitFor(() => document.querySelector(".popover"), 20);
      if (!pop) {
        return {
          reason: "click produced no .popover within 6s",
          chipText: chip.textContent.trim(),
          chipDisabled: chip.disabled === true,
          popoverTotal: document.querySelectorAll(".popover").length,
        };
      }
      const p = pop.getBoundingClientRect();
      const c = chip.getBoundingClientRect();
      const out = {
        cls: pop.className,
        chipText: chip.textContent.trim(),
        below: Math.round(p.top) >= Math.round(c.bottom) - 2,
        above: Math.round(p.bottom) <= Math.round(c.top) + 2,
      };
      document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
      document.body.click();
      await sleep(300);
      return out;
    };

    const emptyPop = await openVariant();
    check("popover: opens down when empty", () => ({
      ok: Boolean(emptyPop && emptyPop.below),
      detail: emptyPop,
    }));

    // ── 5. chat state: export, placement, transcript ────────────────────
    const now = Date.now();
    store.setState({
      messages: [
        { id: "u1", role: "user", content: "What is the capital of France?", createdAt: now - 5000 },
        { id: "a1", role: "assistant", content: "Paris.", createdAt: now - 4000 },
        { id: "u2", role: "user", content: "And of Italy?", createdAt: now - 3000 },
        { id: "a2", role: "assistant", content: "Rome.", createdAt: now - 2000 },
      ],
      streaming: false,
    });
    await sleep(500);

    check("chat: export button visible", () => {
      const btns = Array.from(document.querySelectorAll(".header-actions .hdr-btn"));
      const exportBtn = btns.find((b) => b.getAttribute("aria-label") === "Export conversation");
      return { ok: Boolean(exportBtn) && !exportBtn.disabled, found: Boolean(exportBtn) };
    });

    check("chat: assistant bubbles rendered", () => {
      const n = document.querySelectorAll(".msg.assistant").length;
      return { ok: n === 2, count: n };
    });

    const chatPop = await openVariant();
    check("popover: opens up in a chat", () => ({
      ok: Boolean(chatPop && chatPop.above),
      detail: chatPop,
    }));

    // ── 6. find in conversation ─────────────────────────────────────────
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "f", ctrlKey: true, bubbles: true }));
    await sleep(400);
    const findBar = document.querySelector(".find-bar");
    check("find: bar opens on Ctrl+F", () => ({ ok: Boolean(findBar), found: Boolean(findBar) }));

    if (findBar) {
      const input = findBar.querySelector("input");
      const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
      setter.call(input, "Rome");
      input.dispatchEvent(new Event("input", { bubbles: true }));
      await sleep(500);
      check("find: reports a match", () => {
        const count = document.querySelector(".find-count");
        return { ok: Boolean(count && /^1\/1$/.test(count.textContent.trim())), text: count ? count.textContent.trim() : null };
      });
      document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
      await sleep(250);
      check("find: closes on Escape", () => ({ ok: !document.querySelector(".find-bar") }));
    }

    // ── 7. sibling switcher ─────────────────────────────────────────────
    check("siblings: switcher renders", () => {
      const msg = store.getState().messages[3];
      store.setState({
        messages: store.getState().messages.map((m) =>
          m.id === "a2"
            ? { ...m, siblings: [m, { ...m, id: "a2b", content: "Rome, the capital since 1871." }], siblingIndex: 0 }
            : m,
        ),
      });
      return Boolean(msg);
    });
    await sleep(400);

    check("siblings: switching swaps content", () => {
      const switchers = document.querySelectorAll(".sib-switch");
      if (switchers.length === 0) return { ok: false, reason: "no switcher rendered" };
      const textBefore = document.querySelectorAll(".msg.assistant")[1].textContent;
      store.getState().showSibling("a2", 1);
      return { ok: true, switchers: switchers.length, hadText: textBefore.includes("Rome") };
    });
    await sleep(350);

    check("siblings: content actually changed", () => {
      const text = document.querySelectorAll(".msg.assistant")[1].textContent;
      return { ok: text.includes("1871"), text: text.slice(0, 60) };
    });

    // ── 8. Lottie dots ──────────────────────────────────────────────────
    store.setState({
      messages: [
        { id: "u9", role: "user", content: "Hello", createdAt: Date.now() - 1000 },
        {
          id: "a9", role: "assistant", content: "", createdAt: Date.now(),
          holdMs: 8000, streaming: false,
        },
      ],
      streaming: false,
    });
    await sleep(900);

    check("dots: lottie renders 3 paths", () => {
      const paths = document.querySelectorAll(".tri-dots svg path");
      return { ok: paths.length === 3, count: paths.length };
    });

    check("dots: filled with the accent", () => {
      const p = document.querySelector(".tri-dots svg path");
      if (!p) return { ok: false };
      const fill = getComputedStyle(p).fill;
      return { ok: fill !== "none" && fill !== "rgb(0, 0, 0)", fill };
    });

    check("dots: generating row visible", () => {
      const row = document.querySelector(".gen-row");
      const thought = document.querySelector(".gen-thought");
      return { ok: Boolean(row && thought), thought: thought ? thought.textContent : null };
    });

    // ── 9. glow covers the hold window ──────────────────────────────────
    check("glow: on while holding", () => {
      const main = document.querySelector(".main");
      return { ok: Boolean(main && main.classList.contains("is-glowing")), cls: main ? main.className : null };
    });

    // ── 10. no runtime errors ───────────────────────────────────────────
    check("runtime: no errors thrown", () => ({ ok: errors.length === 0, errors: errors.slice(0, 5) }));

    // reset
    store.getState().newChat();

    const failed = results.filter((r) => !r.ok);
    resolve(
      JSON.stringify(
        {
          passed: results.length - failed.length,
          total: results.length,
          failed: failed.map((f) => f.name),
          results,
        },
        null,
        1,
      ),
    );
  };

  boot().catch((e) => resolve(JSON.stringify({ fatal: String(e && e.stack ? e.stack : e) })));
});
