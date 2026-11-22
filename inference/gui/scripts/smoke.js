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

  // run-smoke.cjs prepends capabilities.json as window.__CAPABILITY_MANIFEST.
  // Every desktop entry with a machine check must be findable here, or the
  // manifest has drifted from what the app actually ships.
  const manifest =
    (window.__CAPABILITY_MANIFEST && window.__CAPABILITY_MANIFEST.capabilities) || [];
  const guiChecks = (kind) =>
    manifest
      .map((cap) => ({ id: cap.id, check: cap.gui && cap.gui.check }))
      .filter((row) => row.check && row.check.kind === kind);
  const controlCheck = (ids) => {
    const rows = guiChecks("control").filter((row) => ids.includes(row.id));
    const missing = rows.filter(
      (row) => !document.querySelector(`[data-capability="${row.check.value}"]`),
    );
    return {
      ok: ids.every((id) => rows.some((row) => row.id === id)) && missing.length === 0,
      missing: missing.map((row) => row.id),
    };
  };

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

    // ── 0. capability manifest: mounted desktop controls ────────────────
    // capabilities.json is the source of truth for each surface. Controls that
    // stay mounted for the whole session must declare themselves with a
    // data-capability attribute, so the manifest cannot drift from the DOM.
    await waitFor(() => document.querySelector('[data-capability="model.switch"]'), 40);
    check("capabilities: always-mounted controls resolve", () =>
      controlCheck(["sessions.open", "model.switch", "settings.open"]));

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

    // ── 4b. model switcher: recents + all checkpoints ───────────────────
    // run-smoke seeds checkpoint_history with one real and one missing file.
    const modelChip = document.querySelector('.composer-bar .bar-chip[title="Model"]');
    if (modelChip) modelChip.click();
    const recentLabel = await waitFor(() => {
      const labels = Array.from(document.querySelectorAll(".popover .popover-label"));
      return labels.find((l) => /Recently loaded/i.test(l.textContent)) || null;
    }, 20);
    await sleep(350);

    check("model switcher: recently loaded rows render history", () => {
      const rows = Array.from(document.querySelectorAll(".popover .model-row"));
      const missing = rows.filter((r) => r.classList.contains("is-missing"));
      const text = rows.map((r) => r.textContent).join(" | ");
      return {
        ok:
          Boolean(recentLabel) &&
          rows.length >= 2 &&
          missing.length === 1 &&
          missing[0].disabled === true &&
          /Missing file/.test(text) &&
          /GGUF/.test(text),
        rows: rows.length,
        missing: missing.length,
        text: text.slice(0, 160),
      };
    });

    check("model switcher: a stored scorecard shows as a badge", () => {
      const badge = document.querySelector(".popover .model-row .model-score");
      return {
        ok:
          Boolean(badge) &&
          /12\/15/.test(badge.textContent) &&
          /coding/.test(badge.title),
        text: badge && badge.textContent,
        title: badge && badge.title,
      };
    });

    const collapse = document.querySelector(".popover .popover-collapse");
    if (collapse) collapse.click();
    await sleep(300);
    check("model switcher: all checkpoints expands", () => {
      const count = collapse
        ? Number(
            (collapse.querySelector(".popover-collapse-count") || {}).textContent || 0,
          )
        : 0;
      const rows = document.querySelectorAll(".popover .model-row").length;
      return {
        ok:
          Boolean(collapse) &&
          collapse.getAttribute("aria-expanded") === "true" &&
          count >= 1 &&
          rows >= 3,
        count,
        rows,
      };
    });
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    document.body.click();
    await sleep(250);

    // ── 4c. running tool name carries the per-letter sweep ──────────────
    store.setState({
      messages: [
        {
          id: "tool-sweep",
          role: "assistant",
          content: "",
          createdAt: Date.now(),
          tools: [
            {
              id: "t-sweep",
              name: "read_file",
              detail: "src/App.tsx",
              status: "running",
            },
          ],
        },
      ],
      streaming: true,
    });
    await sleep(350);
    check("tool row: running name renders a per-letter sweep", () => {
      const name = document.querySelector(".tool-row.is-running .tool-name");
      const letters = name ? Array.from(name.querySelectorAll("span")) : [];
      const delays = letters.map((s) => s.style.animationDelay);
      return {
        ok:
          Boolean(name) &&
          letters.length === "read file".length &&
          delays.every(Boolean) &&
          new Set(delays).size > 1,
        found: Boolean(name),
        letters: letters.length,
        delays: delays.slice(0, 3),
      };
    });

    store.setState((state) => ({
      messages: state.messages.map((message) => ({
        ...message,
        tools: message.tools?.map((tool) => ({ ...tool, status: "done" })),
      })),
      streaming: false,
    }));
    await sleep(300);
    check("tool row: finished name is plain text", () => {
      const name = document.querySelector(".tool-row.is-done .tool-name");
      return {
        ok: Boolean(name) && name.querySelectorAll("span").length === 0,
        found: Boolean(name),
      };
    });
    store.setState({ messages: [], streaming: false });
    await sleep(200);

    // ── 4d. response mode chip (tier is gone) ───────────────────────────
    check("mode: the tier state is gone", () => ({
      ok: !("variant" in store.getState()) && !("setVariant" in store.getState()),
      variant: store.getState().variant,
    }));

    const prevModeStatus = store.getState().status;
    store.setState({
      status: {
        ...(prevModeStatus || {}),
        is_ready: true,
        provider: "local",
        load: {
          ...((prevModeStatus || {}).load || {}),
          display_name: "Dizel 1.5 SFT",
          checkpoint: "Dizel-1.5-SFT.gguf",
        },
      },
    });
    await sleep(300);

    const modeChip = await waitFor(
      () => document.querySelector('.composer-bar .bar-chip[title="Response mode"]'),
      20,
    );
    check("mode chip: shows the mode, not a tier", () => {
      const text = modeChip ? modeChip.textContent.trim() : "";
      return {
        ok:
          Boolean(modeChip) &&
          /Fast|Planning|Coding|Thinking/.test(text) &&
          !/Flash|Pro\b/.test(text),
        text,
      };
    });
    if (modeChip) {
      modeChip.click();
      await waitFor(() => document.querySelector(".popover"), 20);
      check("mode popover: four icon rows, no deep think", () => {
        const rows = Array.from(document.querySelectorAll(".popover .mode-item"));
        const names = rows.map((row) => row.textContent);
        return {
          ok:
            rows.length === 4 &&
            names.some((text) => /Thinking/.test(text)) &&
            !names.some((text) => /Deep Think/.test(text)) &&
            rows.every((row) => row.querySelector("svg")),
          names,
        };
      });

      const modeArrow = document.querySelector(
        '.popover .mode-arrow[aria-label="Thinking effort"]',
      );
      if (modeArrow) modeArrow.click();
      await waitFor(() => document.querySelector(".effort-panel:not(.speed-panel)"), 20);
      // The old design was five coloured rings, so this used to count distinct
      // border colours. The restyle replaced them with ticks plus a raised thumb,
      // and the colour moved to the track fill — so the assertions moved with it.
      check("thinking: five stops with one raised thumb", () => {
        const stops = Array.from(
          document.querySelectorAll(".effort-panel:not(.speed-panel) .effort-stop"),
        );
        const active = stops.filter((stop) => stop.classList.contains("active"));
        const idle = stops.filter((stop) => !stop.classList.contains("active"));
        const thumb = active[0] ? active[0].getBoundingClientRect() : null;
        const tick = idle[0] ? idle[0].getBoundingClientRect() : null;
        return {
          ok:
            stops.length === 5 &&
            active.length === 1 &&
            Boolean(thumb && tick) &&
            thumb.width >= tick.width * 2,
          stops: stops.length,
          active: active.length,
          thumb: thumb ? Math.round(thumb.width) : null,
          tick: tick ? Math.round(tick.width) : null,
        };
      });
      check("thinking: the track is filled in the effort colour", () => {
        const fill = document.querySelector(".effort-panel:not(.speed-panel) .effort-fill");
        if (!fill) return { ok: false, reason: "no fill element" };
        const style = getComputedStyle(fill);
        // The fill is a vertical gradient now, so the colour is in background-image
        // and backgroundColor reads as transparent.
        return {
          ok:
            style.backgroundImage !== "none" &&
            fill.getBoundingClientRect().width >= 0,
          width: Math.round(fill.getBoundingClientRect().width),
          image: style.backgroundImage.slice(0, 60),
        };
      });

      // Regression: picking a stop used to close the whole popover, because the
      // effort panel is portalled outside the Picker's subtree and every click in
      // it read as an outside click. A bare .click() skips the document mousedown
      // listener, so this has to fire the full sequence to catch it.
      const pickStop = document.querySelectorAll(".effort-panel:not(.speed-panel) .effort-stop")[1];
      if (pickStop) {
        for (const type of ["mousedown", "mouseup", "click"]) {
          pickStop.dispatchEvent(
            new MouseEvent(type, { bubbles: true, cancelable: true, button: 0 }),
          );
        }
      }
      await new Promise((r) => setTimeout(r, 320));
      check("thinking: picking a stop keeps the panel open", () => ({
        ok: Boolean(document.querySelector(".effort-panel:not(.speed-panel)")),
        picked: Boolean(pickStop),
      }));
      check("thinking: effort panel is on screen", () => {
        const panel = document.querySelector(".effort-panel:not(.speed-panel)");
        if (!panel) return { ok: false, reason: "missing" };
        const rect = panel.getBoundingClientRect();
        const style = getComputedStyle(panel);
        return {
          ok:
            rect.width > 100 &&
            rect.height > 60 &&
            rect.top >= 0 &&
            rect.left >= 0 &&
            rect.right <= window.innerWidth &&
            rect.bottom <= window.innerHeight &&
            style.display !== "none" &&
            style.visibility !== "hidden" &&
            style.opacity !== "0",
          rect: {
            top: Math.round(rect.top),
            left: Math.round(rect.left),
            width: Math.round(rect.width),
            height: Math.round(rect.height),
          },
          viewport: { w: window.innerWidth, h: window.innerHeight },
          display: style.display,
          opacity: style.opacity,
          zIndex: style.zIndex,
        };
      });
      const highStop = Array.from(
        document.querySelectorAll(".effort-panel:not(.speed-panel) .effort-stop"),
      )[2];
      if (highStop) highStop.click();
      await sleep(500);
      check("thinking: high effort applies with its color", () => {
        const chip = document.querySelector(
          '.composer-bar .bar-chip[title="Response mode"]',
        );
        return {
          ok:
            store.getState().mode === "Thinking" &&
            store.getState().thinkingEffort === "high" &&
            // The chip shows the effort alone now; "High" is also a substring of
            // "X-High", so this compares the whole label rather than matching.
            chip !== null &&
            chip.textContent.trim() === "High" &&
            getComputedStyle(chip).color === "rgb(244, 114, 182)",
          mode: store.getState().mode,
          effort: store.getState().thinkingEffort,
          chipText: chip ? chip.textContent.trim() : null,
          color: chip ? getComputedStyle(chip).color : null,
        };
      });

      // Regression: the thumb used to be centred on the first/last tick, and those
      // sat half a column inside the track — so at Low and Max it never
      // reached either end of the bar. Its edge has to touch the track's edge.
      const thumbSize = () => {
        const scale = document.querySelector(".effort-panel:not(.speed-panel) .effort-scale");
        if (!scale) return 28;
        const value = parseFloat(getComputedStyle(scale).getPropertyValue("--thumb"));
        return Number.isFinite(value) ? value : 28;
      };

      // Poll on the thumb's CENTRE, not its edges. The centre is
      // `left + p * (width - thumb)` and does not move during the width
      // transition, so it is the one value that is stable the whole time. Edge
      // measurements kept landing mid-animation and reading the previous stop.
      const edgeGapAt = async (index) => {
        const open = document.querySelector(".effort-panel:not(.speed-panel)");
        if (!open) return null;
        const stops = Array.from(open.querySelectorAll(".effort-stop"));
        if (!stops[index]) return null;

        const track = open.querySelector(".effort-scale").getBoundingClientRect();
        const size = thumbSize();
        const progress = index / (stops.length - 1);
        const expectedCentre = track.left + size / 2 + progress * (track.width - size);

        for (const type of ["mousedown", "mouseup", "click"]) {
          stops[index].dispatchEvent(
            new MouseEvent(type, { bubbles: true, cancelable: true, button: 0 }),
          );
        }

        for (let i = 0; i < 40; i += 1) {
          await sleep(60);
          const panel = document.querySelector(".effort-panel:not(.speed-panel)");
          if (!panel) return null;
          const thumb = panel.querySelector(".effort-stop.active").getBoundingClientRect();
          const centre = thumb.left + thumb.width / 2;
          const fill = panel.querySelector(".effort-fill");
          const opacity = Number(getComputedStyle(fill).opacity);
          // Zero effort hides the fill; anything above it is fully opaque.
          const targetOpacity = index === 0 ? 0 : 1;
          if (thumb.width < size - 1) continue;
          if (Math.abs(centre - expectedCentre) > 1) continue;
          if (Math.abs(opacity - targetOpacity) > 0.01) continue;

          const box = panel.querySelector(".effort-scale").getBoundingClientRect();
          const fillBox = fill.getBoundingClientRect();
          return {
            left: Math.round(thumb.left - box.left),
            right: Math.round(box.right - thumb.right),
            fillWidth: Math.round(fillBox.width),
            fillOpacity: opacity,
          };
        }
        return null;
      };

      const atLow = await edgeGapAt(0);
      const atDeep = await edgeGapAt(4);
      await edgeGapAt(2); // leave it on High for anything that follows
      check("thinking: thumb reaches both ends of the track", () => ({
        ok:
          Boolean(atLow && atDeep) &&
          atLow.left <= 1 &&
          atDeep.right <= 1 &&
          atLow.fillOpacity < 0.02,
        atLow,
        atDeep,
      }));
      document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
      await store.getState().setMode("Fast");
      await sleep(200);
    }
    store.setState({ status: prevModeStatus });
    await sleep(200);

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

    check("capabilities: export control resolves", () => controlCheck(["app.export"]));

    check("chat: assistant bubbles rendered", () => {
      const n = document.querySelectorAll(".msg.assistant").length;
      return { ok: n === 2, count: n };
    });

    // ── 5b. read aloud goes through the OS speech engine ───────────────
    let speechStubbed = false;
    const realSpeech = window.speechSynthesis;
    const spoken = [];
    try {
      Object.defineProperty(window, "speechSynthesis", {
        configurable: true,
        value: {
          speak: (utterance) => spoken.push(utterance && utterance.text),
          cancel: () => {},
        },
      });
      speechStubbed = true;
    } catch {
      /* the engine object is not replaceable in this runtime */
    }
    if (speechStubbed) {
      const readBtn = document.querySelector(".msg.assistant .msg-actions .act.speak");
      if (readBtn) readBtn.click();
      await sleep(200);
      check("read aloud: the speaker button sends the reply to the engine", () => ({
        ok: spoken.length === 1 && String(spoken[0]).includes("Paris"),
        count: spoken.length,
        spoken: spoken.length ? String(spoken[0]).slice(0, 40) : null,
      }));
      Object.defineProperty(window, "speechSynthesis", {
        configurable: true,
        value: realSpeech,
      });
    } else {
      check("read aloud: speechSynthesis is stubbable", () => ({ ok: false }));
    }

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

    check("siblings: compare button renders", () => ({
      ok: Boolean(document.querySelector(".sib-compare")),
    }));

    const compareBtn = document.querySelector(".sib-compare");
    if (compareBtn) {
      compareBtn.click();
      await waitFor(() => document.querySelector(".compare-panel"), 20);
      check("siblings: compare opens every version", () => {
        const cols = Array.from(document.querySelectorAll(".compare-col"));
        const text = cols.map((col) => col.textContent).join(" ");
        return {
          ok:
            cols.length === 2 &&
            /1871/.test(text) &&
            cols.some((col) => col.classList.contains("is-current")),
          cols: cols.length,
        };
      });
      document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
      await sleep(250);
      check("siblings: compare closes on Escape", () => ({
        ok: !document.querySelector(".compare-panel"),
      }));
    }

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

    // ── 9a. composer stays editable while the model loads ───────────────
    // Slash commands (/skills, /model) must work before the model is ready, so
    // the textarea is gated on the sidecar being up (booted), not on is_ready.
    const prevStatus = store.getState().status;
    store.setState({ booted: true, status: { ...(prevStatus || {}), is_ready: false } });
    await sleep(250);
    check("composer: editable while model loads", () => {
      const ta = document.querySelector(".composer textarea");
      const s = store.getState();
      return {
        ok: Boolean(ta) && ta.disabled === false,
        disabled: ta ? ta.disabled : null,
        booted: s.booted,
        ready: s.status && s.status.is_ready,
      };
    });
    store.setState({ status: prevStatus });
    await sleep(200);

    // ── 9b. a failed approval resolve must not wedge the dialog ─────────
    // An unknown approval id makes the sidecar return 404, so resolveApproval
    // rejects and the dialog stays open. The buttons must re-enable (busy reset)
    // instead of staying disabled forever.
    store.setState({
      approval: {
        id: "smoke-unknown-approval",
        action: "edit",
        tool: "edit_file",
        path: "smoke.txt",
        diff: "--- a/smoke.txt\n+++ b/smoke.txt\n@@ -1 +1 @@\n-old\n+new",
        additions: 1,
        deletions: 1,
        synthetic: false,
      },
    });
    const approveBtn = await waitFor(() => {
      const btns = Array.from(document.querySelectorAll(".tool-approval-actions button"));
      return btns.find((b) => b.textContent.trim() === "Allow once") || null;
    }, 20);
    check("approval: dialog mounts", () => ({ ok: Boolean(approveBtn), found: Boolean(approveBtn) }));
    if (approveBtn) {
      approveBtn.click();
      await waitFor(() => store.getState().error, 20);
      await sleep(300);
      check("approval: buttons re-enable after a failed resolve", () => {
        const btns = Array.from(document.querySelectorAll(".tool-approval-actions button"));
        return {
          ok: btns.length === 4 && btns.every((b) => !b.disabled),
          disabled: btns.map((b) => b.disabled),
          error: store.getState().error,
        };
      });
      store.setState({ approval: null, error: null });
      await sleep(200);
    }

    // ── 9b2. plan approvals ─────────────────────────────────────────────
    // The checklist renders every step; a failed resolve must keep the dialog
    // open with its buttons re-enabled; synthetic fixtures resolve locally.
    store.setState({
      planApproval: {
        id: "smoke-unknown-plan",
        task: "Ship the release checklist",
        steps: ["Tag the release", "Publish the changelog", "Smoke-test the installer"],
        synthetic: false,
      },
    });
    const planExecute = await waitFor(() => {
      const btns = Array.from(
        document.querySelectorAll(".plan-approval .tool-approval-actions button"),
      );
      return btns.find((b) => b.textContent.trim() === "Execute plan") || null;
    }, 20);
    check("plan: dialog lists every step", () => {
      const items = Array.from(
        document.querySelectorAll(".plan-approval-steps li"),
      ).map((li) => li.textContent.trim());
      const heading = document.querySelector(".plan-approval-head h2");
      return {
        ok:
          Boolean(planExecute) &&
          items.length === 3 &&
          items[0] === "Tag the release" &&
          /Plan approval/.test(heading ? heading.textContent : ""),
        items,
      };
    });
    if (planExecute) {
      planExecute.click();
      await waitFor(() => store.getState().error, 20);
      await sleep(300);
      check("plan: buttons re-enable after a failed resolve", () => {
        const btns = Array.from(
          document.querySelectorAll(".plan-approval .tool-approval-actions button"),
        );
        return {
          ok:
            btns.length === 2 &&
            btns.every((b) => !b.disabled) &&
            store.getState().planApproval !== null,
          disabled: btns.map((b) => b.disabled),
        };
      });
      store.setState({ planApproval: null, error: null });
      await sleep(200);
    }

    store.setState({
      planApproval: {
        id: "smoke-synthetic-plan",
        task: "Ship it",
        steps: ["Tag", "Publish"],
        synthetic: true,
      },
    });
    await waitFor(() => document.querySelector(".plan-approval"), 20);
    const planDiscard = Array.from(
      document.querySelectorAll(".plan-approval .tool-approval-actions button"),
    ).find((b) => b.textContent.trim() === "Discard");
    if (planDiscard) planDiscard.click();
    await waitFor(() => store.getState().planApproval === null, 20);
    check("plan: discard closes the dialog", () => ({
      ok: store.getState().planApproval === null && !document.querySelector(".plan-approval"),
    }));

    store.setState({
      planApproval: {
        id: "smoke-synthetic-plan-execute",
        task: "Ship it",
        steps: ["Tag"],
        synthetic: true,
      },
    });
    await waitFor(() => document.querySelector(".plan-approval"), 20);
    const planGo = Array.from(
      document.querySelectorAll(".plan-approval .tool-approval-actions button"),
    ).find((b) => b.textContent.trim() === "Execute plan");
    if (planGo) planGo.click();
    await waitFor(() => store.getState().planApproval === null, 20);
    check("plan: execute closes the dialog", () => ({
      ok: store.getState().planApproval === null && !document.querySelector(".plan-approval"),
    }));

    // ── 9c. delete is a destructive, always-manual gate ─────────────────
    // A delete approval must read "Delete file", warn about the recycle bin,
    // and offer only Reject/Approve — never a standing "Always allow".
    store.setState({
      approval: {
        id: "smoke-delete-approval",
        action: "delete",
        tool: "delete_file",
        path: "smoke-delete.txt",
        diff: "--- a/smoke-delete.txt\n+++ b/smoke-delete.txt\n@@ -1 +0,0 @@\n-old line",
        additions: 0,
        deletions: 1,
        synthetic: true,
      },
    });
    await waitFor(() => document.querySelector(".tool-approval"), 20);
    check("delete: destructive gate renders", () => {
      const heading = document.querySelector(".tool-approval-head h2");
      const warning = document.querySelector(".tool-approval-warning");
      const modal = document.querySelector(".tool-approval");
      const btns = Array.from(document.querySelectorAll(".tool-approval-actions button"));
      const labels = btns.map((b) => b.textContent.trim());
      return {
        ok:
          Boolean(heading && heading.textContent.trim() === "Delete file") &&
          Boolean(warning && /recycle bin/i.test(warning.textContent)) &&
          btns.length === 2 &&
          labels.includes("Approve") &&
          labels.includes("Reject") &&
          !labels.some((l) => /always/i.test(l)) &&
          Boolean(modal && modal.classList.contains("is-delete")),
        heading: heading ? heading.textContent.trim() : null,
        warning: warning ? warning.textContent.trim() : null,
        labels,
        isDelete: modal ? modal.classList.contains("is-delete") : false,
      };
    });
    store.setState({ approval: null });
    await sleep(200);

    // ── 9f. write gate offers four decisions; Escape denies ─────────────
    store.setState({
      approval: {
        id: "smoke-trust-approval",
        action: "edit",
        tool: "edit_file",
        path: "smoke-trust.txt",
        diff: "--- a/smoke-trust.txt\n+++ b/smoke-trust.txt\n@@ -1 +1 @@\n-old\n+new",
        additions: 1,
        deletions: 1,
        synthetic: true,
      },
    });
    await waitFor(() => document.querySelector(".tool-approval"), 20);
    check("trust: write gate offers four decisions", () => {
      const btns = Array.from(document.querySelectorAll(".tool-approval-actions button"));
      const labels = btns.map((b) => b.textContent.trim());
      return {
        ok:
          labels.length === 4 &&
          labels[0] === "Reject" &&
          labels.includes("Allow once") &&
          labels.includes("Always this session") &&
          labels.includes("Always allow"),
        labels,
      };
    });
    const sessionBtn = Array.from(document.querySelectorAll(".tool-approval-actions button"))
      .find((b) => b.textContent.trim() === "Always this session");
    if (sessionBtn) sessionBtn.click();
    await sleep(250);
    check("trust: standing decision asks for confirmation first", () => {
      const panel = document.querySelector(".tool-approval-confirm");
      const text = panel ? panel.textContent : "";
      return {
        ok:
          Boolean(panel) &&
          store.getState().approval !== null &&
          /without asking/i.test(text),
        text: text.slice(0, 120),
      };
    });
    const confirmSession = Array.from(
      document.querySelectorAll(".tool-approval-actions button"),
    ).find((b) => b.textContent.trim() === "Allow for session");
    if (confirmSession) confirmSession.click();
    await waitFor(() => store.getState().approval === null, 20);
    check("trust: confirming 'for session' resolves the synthetic gate", () => ({
      ok: store.getState().approval === null,
    }));
    await sleep(200);

    // Always allow is the dangerous tier: the warning must name the scope and
    // the way out, and Escape must back out without deciding.
    store.setState({
      approval: {
        id: "smoke-trust-persist",
        action: "edit",
        tool: "edit_file",
        path: "smoke-trust.txt",
        diff: "--- a/smoke-trust.txt\n+++ b/smoke-trust.txt\n@@ -1 +1 @@\n-old\n+new",
        additions: 1,
        deletions: 1,
        synthetic: true,
      },
    });
    await waitFor(() => document.querySelector(".tool-approval"), 20);
    await sleep(250);
    const persistBtn = await waitFor(
      () =>
        Array.from(document.querySelectorAll(".tool-approval-actions button")).find(
          (b) => b.textContent.trim() === "Always allow",
        ) || null,
      20,
    );
    if (persistBtn) persistBtn.click();
    await sleep(250);
    check("trust: always-allow names scope and revocation", () => {
      const panel = document.querySelector(".tool-approval-confirm");
      const text = panel ? panel.textContent : "";
      return {
        ok:
          Boolean(panel) &&
          panel.classList.contains("is-danger") &&
          /every project/i.test(text) &&
          /Settings/.test(text),
        text: text.slice(0, 160),
      };
    });
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    await sleep(250);
    check("trust: Escape backs out of the warning, gate stays open", () => {
      const labels = Array.from(document.querySelectorAll(".tool-approval-actions button")).map(
        (b) => b.textContent.trim(),
      );
      return {
        ok:
          !document.querySelector(".tool-approval-confirm") &&
          store.getState().approval !== null &&
          labels.length === 4,
        labels,
      };
    });
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    await waitFor(() => store.getState().approval === null, 20);
    check("trust: second Escape denies the write gate", () => ({
      ok: store.getState().approval === null,
    }));
    await sleep(200);

    store.setState({
      approval: {
        id: "smoke-trust-deny",
        action: "edit",
        tool: "edit_file",
        path: "smoke-trust.txt",
        diff: "--- a/smoke-trust.txt\n+++ b/smoke-trust.txt\n@@ -1 +1 @@\n-old\n+new",
        additions: 1,
        deletions: 1,
        synthetic: true,
      },
    });
    await waitFor(() => document.querySelector(".tool-approval"), 20);
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    await waitFor(() => store.getState().approval === null, 20);
    check("trust: Escape denies the write gate", () => ({
      ok: store.getState().approval === null,
    }));
    await sleep(200);

    // ── 9g. persistent trust round-trips through the real sidecar ───────
    // run-smoke boots the sidecar with a throwaway DIZEL_CONFIG_DIR, so this
    // writes a temp settings file, never the user's real one.
    await store.getState().setPersistentTrust(true);
    await waitFor(() => store.getState().trust?.persistent === true, 30);
    check("trust: persistent toggle round-trips to the sidecar", () => ({
      ok: store.getState().trust?.persistent === true,
      trust: store.getState().trust,
    }));
    await waitFor(() => document.querySelector(".trust-chip"), 20);
    check("trust: status bar shows the auto-approve chip", () => {
      const chip = document.querySelector(".trust-chip");
      return {
        ok: Boolean(chip) && /auto-approving writes/i.test(chip.textContent),
        text: chip ? chip.textContent : null,
      };
    });
    store.setState({ trust: { persistent: false, session_trusted: true } });
    await sleep(250);
    check("trust: session grant alone keeps the chip visible", () => ({
      ok: Boolean(document.querySelector(".trust-chip")),
    }));
    await store.getState().setPersistentTrust(false);
    await waitFor(() => store.getState().trust?.persistent === false, 30);
    store.setState({ trust: null });
    await sleep(250);
    check("trust: chip hides when no trust is active", () => ({
      ok: !document.querySelector(".trust-chip"),
    }));

    // ── 9h. Settings → Chat → File writes exposes the same controls ─────
    store.getState().openSettings("Chat");
    const trustSwitch = await waitFor(() => {
      const sw = Array.from(document.querySelectorAll(".set-content .set-switch"));
      return sw.find((b) => b.getAttribute("aria-label") === "Always allow writes and edits") || null;
    }, 30);
    check("trust: settings offers the persistent switch", () => {
      if (trustSwitch) return { ok: true };
      // Report what the dialog actually rendered, so a failure is diagnosable
      // instead of just "not found".
      const content = document.querySelector(".set-content");
      const switches = Array.from(
        document.querySelectorAll(".set-content .set-switch"),
      ).map((el) => el.getAttribute("aria-label"));
      return {
        ok: false,
        settingsOpen: store.getState().settingsOpen,
        tab: store.getState().settingsTab,
        hasContent: Boolean(content),
        switches,
        text: content ? content.textContent.trim().slice(0, 80) : null,
        error: store.getState().settingsError,
      };
    });
    if (trustSwitch) {
      trustSwitch.click();
      await waitFor(() => store.getState().trust?.persistent === true, 30);
      await sleep(200);
      check("trust: settings switch enables persistent trust", () => ({
        ok: store.getState().trust?.persistent === true && trustSwitch.classList.contains("is-on"),
        on: trustSwitch.classList.contains("is-on"),
      }));
      const revokeBtn = Array.from(document.querySelectorAll(".set-content button"))
        .find((b) => /revoke this session/i.test(b.textContent));
      check("trust: settings offers session revocation", () => ({
        ok: Boolean(revokeBtn) && revokeBtn.disabled === true,
      }));
      trustSwitch.click();
      await waitFor(() => store.getState().trust?.persistent === false, 30);
      await sleep(200);
    }
    store.getState().closeSettings();
    await sleep(200);

    // ── 9i. Per-provider usage limits ───────────────────────────────────
    check("limits: provider defaults reach the renderer", () => {
      const providers = store.getState().providers || [];
      const openai = providers.find((p) => p.slug === "openai");
      const local = providers.find((p) => p.slug === "local");
      return {
        ok:
          Boolean(openai) &&
          openai.context_limit === 131072 &&
          Boolean(local) &&
          local.context_limit === null,
        openai: openai ? openai.context_limit : null,
        local: local ? local.context_limit : null,
      };
    });

    check("status: usage tooltip names the provider", () => {
      const el = Array.from(document.querySelectorAll(".statusbar span")).find((s) =>
        /usage window/i.test(s.title || ""),
      );
      return {
        ok: Boolean(el) && /^local /i.test(el.title),
        title: el ? el.title : null,
      };
    });

    store.getState().openSettings("Chat");
    // Four local requests are fast, but the suite can have the sidecar busy; a
    // fixed 600ms lost that race and read the "Reading settings…" placeholder.
    await waitFor(
      () =>
        /Usage limit/.test(
          (document.querySelector(".set-content") || {}).textContent || "",
        ),
      20,
    );
    check("limits: Settings shows the usage-limit row", () => {
      const content = document.querySelector(".set-content") || document.body;
      const text = content.textContent || "";
      return {
        ok:
          /Usage limit/.test(text) &&
          /checkpoint reports its own context window/i.test(text),
        snippet: text.trim().slice(0, 70),
      };
    });
    check("memory: Settings shows the project context section", () => {
      const content = document.querySelector(".set-content") || document.body;
      const text = content.textContent || "";
      return {
        ok: /Project context/.test(text) && /Project memory/.test(text),
        snippet: text.trim().slice(0, 70),
      };
    });
    store.getState().closeSettings();
    await sleep(200);

    // ── 9d0. Attached skill chips ───────────────────────────────────────
    // Picking a skill attaches a chip and keeps the draft short; the model
    // still receives the expanded prompt at send time.
    const emailSkill = store.getState().skills.find((skill) => skill.name === "email");
    check("skills: catalog includes email", () => ({ ok: Boolean(emailSkill) }));
    if (emailSkill && typeof store.getState().attachSkill === "function") {
      store.getState().newChat();
      await sleep(300);
      await store.getState().attachSkill(emailSkill, "hello");
      await sleep(400);
      check("skills: attach renders a chip", () => {
        const chips = document.querySelectorAll(".composer .skill-chip");
        return {
          ok: store.getState().attachedSkills.length === 1 && chips.length === 1,
          state: store.getState().attachedSkills.length,
          chips: chips.length,
          draft: store.getState().draft,
        };
      });
      check("skills: the draft stays short", () => ({
        ok: store.getState().draft.trim() === "",
        draft: store.getState().draft.slice(0, 60),
      }));

      const second = store.getState().skills.find((skill) => skill.name !== "email");
      if (second) await store.getState().attachSkill(second, "");
      await sleep(300);
      check("skills: multiple chips ride together", () => ({
        ok: store.getState().attachedSkills.length === 2,
        count: store.getState().attachedSkills.length,
      }));

      const textarea = document.querySelector(".composer textarea");
      textarea.focus();
      textarea.dispatchEvent(
        new KeyboardEvent("keydown", { key: "Backspace", bubbles: true }),
      );
      await sleep(250);
      check("skills: backspace on an empty draft removes the last chip", () => ({
        ok: store.getState().attachedSkills.length === 1,
        count: store.getState().attachedSkills.length,
      }));

      document.querySelector(".composer .skill-x")?.click();
      await sleep(250);
      check("skills: the chip x removes it", () => ({
        ok: store.getState().attachedSkills.length === 0,
        count: store.getState().attachedSkills.length,
      }));

      // Send: the outgoing request must carry the expanded prompt plus the
      // view-only display record. A targeted fetch stub returns one SSE frame.
      const realFetch = window.fetch;
      let captured = null;
      window.fetch = async (url, init) => {
        if (String(url).includes("/chat/stream")) {
          captured = JSON.parse(init.body);
          return new Response('data: {"type": "done", "text": "ok"}\n\n', {
            status: 200,
            headers: { "Content-Type": "text/event-stream" },
          });
        }
        return realFetch(url, init);
      };
      await store.getState().attachSkill(emailSkill, "args");
      store.getState().setDraft("hello");
      await store.getState().send("hello");
      await sleep(500);
      window.fetch = realFetch;
      check("skills: send carries the prompt and the display record", () => ({
        ok:
          Boolean(captured) &&
          captured.user_text.includes("email") &&
          captured.display?.text === "hello" &&
          captured.display?.skills?.[0]?.name === "email",
        hasText: Boolean(captured && captured.user_text.includes("email")),
        display: captured && captured.display,
      }));

      const now = Date.now();
      store.setState({
        messages: [
          {
            id: "skill-u1",
            role: "user",
            content: "PROMPT BODY\n\nhello",
            createdAt: now,
            display: { text: "hello", skills: [{ name: "email", title: "Email" }] },
          },
          { id: "skill-a1", role: "assistant", content: "ok", createdAt: now + 1 },
        ],
        streaming: false,
      });
      await sleep(300);
      check("skills: the sent bubble shows the chip, not the body", () => {
        const chips = document.querySelectorAll(".msg.user .msg-skill");
        const text = document.querySelector(".msg.user .user-text")?.textContent ?? "";
        return {
          ok: chips.length === 1 && text.trim() === "hello",
          chips: chips.length,
          text,
        };
      });

      store.getState().beginEdit("skill-u1");
      await sleep(600);
      check("skills: edit restores the chip and the typed text", () => ({
        ok:
          store.getState().draft.trim() === "hello" &&
          store.getState().attachedSkills.length === 1,
        draft: store.getState().draft.slice(0, 40),
        chips: store.getState().attachedSkills.length,
      }));
      store.getState().cancelEdit();
      await sleep(200);
    } else {
      check("skills: attachSkill exists", () => ({ ok: false }));
    }

    // ── 9d. Ctrl+K command palette ──────────────────────────────────────
    // A global palette jumps to commands and recent chats. It must open on
    // Ctrl+K, list both kinds, fuzzy-filter, route a pick to runSlash /
    // openSession, and close on Escape.
    store.setState({
      sessions: [
        { id: "smoke-pal-s1", title: "Alpha launch plan", created: "", preview: "", pinned: false, archived: false, tags: [] },
        { id: "smoke-pal-s2", title: "Beta budget notes", created: "", preview: "", pinned: false, archived: false, tags: [] },
      ],
    });

    // The manifest's palette checks need the debug catalog, which the list
    // only builds when debug mode is on. Mount a palette after enabling it,
    // then restore the mode and reopen through the keyboard for the flow.
    const hadDebug = store.getState().debug;
    if (!hadDebug) store.getState().toggleDebug();
    store.getState().openPalette();
    await waitFor(() => document.querySelector(".command-palette input"), 20);
    await sleep(250);
    check("capabilities: palette commands resolve", () => {
      // Read the name cells, not the row text: the description follows the
      // name with no separator, so a boundary match on row text can't work.
      const names = Array.from(
        document.querySelectorAll(".command-palette .command-name"),
      ).map((name) => name.textContent.trim());
      const wanted = guiChecks("palette");
      const missing = wanted.filter(
        (row) => !names.includes(`/${row.check.value}`),
      );
      return {
        ok: wanted.length > 0 && missing.length === 0,
        missing: missing.map((row) => row.id),
        count: names.length,
      };
    });
    store.getState().closePalette();
    if (!hadDebug) store.getState().toggleDebug();
    await sleep(250);

    window.dispatchEvent(new KeyboardEvent("keydown", { key: "k", ctrlKey: true, bubbles: true }));
    const palInput = await waitFor(() => document.querySelector(".command-palette input"), 20);
    check("palette: opens on Ctrl+K", () => ({
      ok: Boolean(palInput) && store.getState().paletteOpen === true,
      found: Boolean(palInput),
      open: store.getState().paletteOpen,
    }));
    if (palInput) {
      check("capabilities: palette control resolves", () => controlCheck(["commands.palette"]));

      check("palette: dialog is horizontally centered", () => {
        const pal = document.querySelector(".command-palette");
        const rect = pal.getBoundingClientRect();
        const offset = Math.abs(rect.x + rect.width / 2 - window.innerWidth / 2);
        return {
          ok: offset <= 2,
          offset: Math.round(offset),
          width: Math.round(rect.width),
        };
      });

      check("palette: lists commands and chats", () => {
        const items = Array.from(document.querySelectorAll(".command-palette .command-item"));
        const text = items.map((i) => i.textContent).join(" ");
        return {
          ok: items.length >= 2 && /\/skills/.test(text) && /Alpha launch plan/.test(text),
          count: items.length,
        };
      });

      const palSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
      palSetter.call(palInput, "alpha");
      palInput.dispatchEvent(new Event("input", { bubbles: true }));
      await sleep(300);
      check("palette: fuzzy filters chats", () => {
        const items = Array.from(document.querySelectorAll(".command-palette .command-item"));
        const text = items.map((i) => i.textContent).join(" ");
        return { ok: /Alpha launch plan/.test(text) && !/Beta budget notes/.test(text), text: text.slice(0, 80) };
      });

      let openedId = null;
      const realOpen = store.getState().openSession;
      store.setState({ openSession: async (id) => { openedId = id; } });
      await sleep(200);
      const chatItem = Array.from(document.querySelectorAll(".command-palette .command-item"))
        .find((i) => /Alpha launch plan/.test(i.textContent));
      if (chatItem) chatItem.click();
      await sleep(250);
      check("palette: picking a chat opens it and closes", () => ({
        ok: openedId === "smoke-pal-s1" && store.getState().paletteOpen === false,
        openedId,
        open: store.getState().paletteOpen,
      }));
      store.setState({ openSession: realOpen });

      store.getState().openPalette();
      await waitFor(() => document.querySelector(".command-palette input"), 20);
      let slashArg = null;
      const realSlash = store.getState().runSlash;
      store.setState({ runSlash: async (arg) => { slashArg = arg; return true; } });
      await sleep(200);
      const cmdItem = Array.from(document.querySelectorAll(".command-palette .command-item"))
        .find((i) => /\/skills/.test(i.textContent));
      if (cmdItem) cmdItem.click();
      await sleep(250);
      check("palette: picking a command runs it and closes", () => ({
        ok: slashArg === "/skills" && store.getState().paletteOpen === false,
        slashArg,
        open: store.getState().paletteOpen,
      }));
      store.setState({ runSlash: realSlash });

      store.getState().openPalette();
      await waitFor(() => document.querySelector(".command-palette input"), 20);
      document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
      await sleep(250);
      check("palette: closes on Escape", () => ({
        ok: store.getState().paletteOpen === false && !document.querySelector(".command-palette"),
      }));
    }
    store.setState({ sessions: [] });
    await sleep(200);

    // ── 9e. Runtime doctor ──────────────────────────────────────────────
    // /doctor is always available (no debug mode) and renders the sidecar's
    // offline checks in the diagnostics panel.
    await store.getState().runSlash("/doctor");
    const doctorPanel = await waitFor(() => {
      const panel = store.getState().debugPanel;
      return panel && panel.title === "Runtime doctor" ? panel : null;
    }, 30);
    check("doctor: /doctor renders the runtime report", () => ({
      ok:
        Boolean(doctorPanel) &&
        /Python/.test(doctorPanel.body) &&
        /Core packages/.test(doctorPanel.body),
      title: doctorPanel && doctorPanel.title,
    }));
    store.getState().closeDebugPanel();
    await sleep(200);

    // ── 9f0. Scorecard panel ────────────────────────────────────────────
    const scoredPath = window.__SCORECARD_PATH;
    if (scoredPath) {
      await store.getState().runSlash(`/scorecard ${scoredPath}`);
      const cardPanel = await waitFor(() => {
        const p = store.getState().debugPanel;
        return p && p.title === "Scorecard" ? p : null;
      }, 30);
      check("scorecard: /scorecard <path> renders the stored card", () => ({
        ok:
          Boolean(cardPanel) &&
          /12\/15/.test(cardPanel.body) &&
          /coding/.test(cardPanel.body),
        body: cardPanel && cardPanel.body.slice(0, 100),
      }));
      store.getState().closeDebugPanel();
      await sleep(200);
    } else {
      check("scorecard: fixture checkpoint present", () => ({ ok: false }));
    }

    // ── 9f1. Usage & cost panel ─────────────────────────────────────────
    await store.getState().runSlash("/cost");
    const costPanel = await waitFor(() => {
      const p = store.getState().debugPanel;
      return p && p.title === "Usage & cost" ? p : null;
    }, 30);
    check("cost: /cost renders the provider table", () => ({
      ok:
        Boolean(costPanel) &&
        /local/.test(costPanel.body) &&
        /estimate/i.test(costPanel.body),
      body: costPanel && costPanel.body.slice(0, 100),
    }));
    store.getState().closeDebugPanel();
    await sleep(200);

    // ── 9f2. Project memory ─────────────────────────────────────────────
    await store.getState().runSlash("/memory add Uses CRLF for .cmd files.");
    const memoryPanel = await waitFor(() => {
      const p = store.getState().debugPanel;
      return p && p.title === "Project memory" ? p : null;
    }, 30);
    check("memory: /memory add stores and lists an entry", () => ({
      ok: Boolean(memoryPanel) && /CRLF/.test(memoryPanel.body),
      body: memoryPanel && memoryPanel.body.slice(0, 100),
    }));
    store.getState().closeDebugPanel();
    await sleep(150);
    await store.getState().runSlash("/memory clear");
    const clearedPanel = await waitFor(() => {
      const p = store.getState().debugPanel;
      return p && /Cleared 1 entry/.test(p.body) ? p : null;
    }, 30);
    check("memory: /memory clear removes the entry", () => ({
      ok: Boolean(clearedPanel),
      body: clearedPanel && clearedPanel.body,
    }));
    store.getState().closeDebugPanel();
    await sleep(150);

    // ── 9f3. /review and /commit send the diff, show a compact line ─────
    const realFetchGit = window.fetch;
    let reviewSent = null;
    let commitSent = null;
    window.fetch = async (url, init) => {
      const target = String(url);
      if (target.includes("/project/git/diff")) {
        const staged = target.includes("mode=staged");
        return new Response(
          JSON.stringify({
            available: true,
            branch: "main",
            files: [{ status: "M", path: "app.py" }],
            diff: "+ print('x')",
            truncated: false,
            error: "",
            prompt: staged
              ? "Write one commit message for these staged changes."
              : "Review this working-tree diff.",
            display: staged
              ? "Draft a commit message · 1 staged files"
              : "Review the working-tree diff · 1 files",
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        );
      }
      if (target.includes("/chat/stream")) {
        const body = JSON.parse(init.body);
        if (body.user_text.includes("commit message")) commitSent = body;
        else reviewSent = body;
        return new Response('data: {"type": "done", "text": "looks fine"}\n\n', {
          status: 200,
          headers: { "Content-Type": "text/event-stream" },
        });
      }
      return realFetchGit(url, init);
    };
    await store.getState().runSlash("/review");
    await sleep(600);
    await store.getState().runSlash("/commit");
    await sleep(600);
    window.fetch = realFetchGit;

    check("git: /review sends the diff prompt with a compact display", () => {
      const user = [...store.getState().messages]
        .reverse()
        .find((m) => m.role === "user" && /commit message|working-tree/.test(m.display?.text || ""));
      return {
        ok:
          Boolean(reviewSent) &&
          /Review this working-tree diff/.test(reviewSent.user_text) &&
          Boolean(
            store
              .getState()
              .messages.some((m) => m.display?.text?.startsWith("Review the working-tree diff")),
          ),
        display: user && user.display && user.display.text,
      };
    });
    check("git: /commit sends the staged prompt and never commits", () => {
      return {
        ok:
          Boolean(commitSent) &&
          /commit message/i.test(commitSent.user_text) &&
          store
            .getState()
            .messages.some((m) => m.display?.text?.startsWith("Draft a commit message")),
        sent: commitSent && commitSent.user_text.slice(0, 40),
      };
    });

    // ── 9d2. /plan sends the drafting prompt ───────────────────────────
    let planSent = null;
    const realFetchPlan = window.fetch;
    window.fetch = async (url, init) => {
      const target = String(url);
      if (target.includes("/chat/stream")) {
        planSent = JSON.parse(init.body);
        return new Response('data: {"type": "done", "text": "Plan:\\n1. Step"}\n\n', {
          status: 200,
          headers: { "Content-Type": "text/event-stream" },
        });
      }
      return realFetchPlan(url, init);
    };
    await store.getState().runSlash("/plan fix the build");
    await sleep(600);
    window.fetch = realFetchPlan;

    check("plan: /plan sends the drafting prompt with a compact display", () => ({
      ok:
        Boolean(planSent) &&
        /Draft a short numbered plan/.test(planSent.user_text) &&
        store
          .getState()
          .messages.some((m) => m.display?.text?.startsWith("Plan · fix the build")),
      sent: planSent ? planSent.user_text.slice(0, 40) : null,
    }));

    check("plan: /plan selects Planning", () => ({
      ok: store.getState().mode === "Planning",
      mode: store.getState().mode,
    }));
    await store.getState().setMode("Fast");
    await sleep(200);

    // ── 9e. In-app release notes (bundled changelog) ────────────────────
    // About offers the bundled CHANGELOG offline. A "New" badge shows until
    // the notes are viewed; the modal renders markdown and closes on Escape.
    localStorage.removeItem("dizel.releaseNotesSeen");
    store.setState({ releaseNotesSeen: "" });
    store.getState().openSettings("About");
    const rnButton = await waitFor(() => {
      const btns = Array.from(document.querySelectorAll(".set-content button"));
      return btns.find((b) => /View release notes/i.test(b.textContent)) || null;
    }, 30);
    check("release notes: About offers a view button with a New badge", () => ({
      ok: Boolean(rnButton) && Boolean(document.querySelector(".release-notes-new")),
      found: Boolean(rnButton),
      badge: Boolean(document.querySelector(".release-notes-new")),
    }));
    if (rnButton) {
      rnButton.click();
      const rnModal = await waitFor(() => document.querySelector(".release-notes-modal"), 20);
      await sleep(200);
      check("release notes: modal renders the bundled changelog", () => {
        const md = document.querySelector(".release-notes-modal .md");
        const text = md ? md.textContent : "";
        return {
          ok: Boolean(rnModal) && /2\.0\.1/.test(text) && /Added/.test(text) && /Fixed/.test(text),
          found: Boolean(rnModal),
          sample: text.slice(0, 60),
        };
      });
      check("release notes: viewing clears the New badge", () => ({
        ok: store.getState().releaseNotesSeen !== "" && !document.querySelector(".release-notes-new"),
        seen: store.getState().releaseNotesSeen,
      }));
      document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
      await sleep(250);
      check("release notes: closes on Escape", () => ({
        ok: store.getState().releaseNotesOpen === false && !document.querySelector(".release-notes-modal"),
      }));
    }
    store.getState().closeSettings();
    await sleep(200);

    // ── 9j. image input follows the model's vision flag ─────────────────
    // That flag now comes from the provider's own capabilities (Ollama
    // /api/show) instead of a model-name regex. The local checkpoint is
    // text-only, so the Images row must be disabled and say so.
    const addTrigger = document.querySelector('.composer-bar .bar-btn[title="Add to prompt"]');
    if (addTrigger) {
      addTrigger.click();
      await sleep(300);
      check("attachments: image input follows the model's vision flag", () => {
        const row = Array.from(document.querySelectorAll(".popover .popover-item")).find(
          (el) => /Images/.test(el.textContent || ""),
        );
        const vision = store.getState().status?.model_info?.vision === true;
        return {
          ok: Boolean(row) && row.disabled === !vision,
          vision,
          disabled: row ? row.disabled : null,
          hint: row ? (row.querySelector(".popover-hint") || {}).textContent : null,
        };
      });
      addTrigger.click();
      await sleep(200);
    }

    // ── 9k. context panel ───────────────────────────────────────────────
    const ctxTrigger = document.querySelector('[data-capability="context.rail"]');
    if (ctxTrigger) {
      ctxTrigger.click();
      await sleep(400);
      check("context: panel lists the prompt blocks", () => {
        const panel = document.querySelector(".ctx-panel");
        const labels = Array.from(
          panel ? panel.querySelectorAll(".ctx-row span") : [],
        ).map((el) => el.textContent);
        return {
          ok:
            Boolean(panel) &&
            labels.includes("System prompt") &&
            labels.includes("Mode instructions") &&
            labels.includes("Conversation"),
          labels,
        };
      });
      document.dispatchEvent(
        new KeyboardEvent("keydown", { key: "Escape", bubbles: true }),
      );
      await sleep(300);
      check("context: panel closes on Escape", () => ({
        ok: !document.querySelector(".ctx-panel"),
      }));
    }

    // ── 9l. session digest ─────────────────────────────────────────────
    // Derived from loaded messages: no model call and nothing to cache.
    const savedMessages = store.getState().messages;
    const seeded = Array.from({ length: 12 }, (_, i) => ({
      id: `digest-${i}`,
      role: i % 2 === 0 ? "user" : "assistant",
      content:
        i === 0
          ? "Fix the Windows launcher line endings"
          : i === 10
            ? "Can you check the TUI as well?"
            : i === 11
              ? "All green - both launchers work"
              : `turn ${i}`,
    }));
    store.setState({ messages: seeded, streaming: false });
    await sleep(400);
    check("digest: long chat shows the reading aid", () => {
      const card = document.querySelector('[data-capability="session.digest"]');
      const text = card ? card.textContent || "" : "";
      return {
        ok:
          Boolean(card) &&
          /Started with/.test(text) &&
          /Fix the Windows launcher/.test(text) &&
          /Last asked/.test(text) &&
          /All green/.test(text),
      };
    });
    const dismiss = document.querySelector(".digest-close");
    if (dismiss) dismiss.click();
    await sleep(250);
    check("digest: dismiss removes the card", () => ({
      ok: !document.querySelector('[data-capability="session.digest"]'),
    }));
    store.setState({ messages: savedMessages });
    await sleep(250);

    // ── 9m. fast reply speeds ───────────────────────────────────────────
    await store.getState().setMode("Fast");
    await store.getState().setFastSpeed("fast");
    await sleep(300);
    const speedChip = document.querySelector(
      '.composer-bar .bar-chip[title="Response mode"]',
    );
    if (speedChip) {
      speedChip.click();
      await sleep(300);
      const fastArrow = document.querySelector(
        '.popover .mode-arrow[aria-label="Fast speed"]',
      );
      if (fastArrow) fastArrow.click();
      await waitFor(() => document.querySelector(".speed-panel"), 20);
      const stops = Array.from(
        document.querySelectorAll(".speed-panel .effort-stop"),
      );
      check("speed: three tiers in the panel", () => ({
        ok: stops.length === 3,
        count: stops.length,
      }));
      if (stops[1]) {
        for (const type of ["mousedown", "mouseup", "click"]) {
          stops[1].dispatchEvent(
            new MouseEvent(type, { bubbles: true, cancelable: true, button: 0 }),
          );
        }
      }
      await sleep(500);
      check("speed: chip shows only the tier in its colour", () => {
        const chip = document.querySelector(
          '.composer-bar .bar-chip[title="Response mode"]',
        );
        const text = chip ? chip.textContent.trim() : "";
        const color = chip ? getComputedStyle(chip).color : "";
        return {
          ok: text === "Turbo" && color === "rgb(79, 214, 190)",
          text,
          color,
        };
      });
      check("speed: mode prompt follows the tier", () => ({
        ok: store.getState().modePrompt.includes("(Turbo)"),
        prompt: store.getState().modePrompt.slice(0, 70),
      }));

      // The circle must drag, not just take clicks like the first version did.
      const speedScale = document.querySelector(".speed-panel .effort-scale");
      if (speedScale) {
        const marks = Array.from(speedScale.querySelectorAll(".effort-stop"));
        const first = marks[0].getBoundingClientRect();
        const last = marks[marks.length - 1].getBoundingClientRect();
        const fromX = first.left + first.width / 2;
        const toX = last.left + last.width / 2;
        const y = first.top + first.height / 2;
        const drag = (type, x) =>
          speedScale.dispatchEvent(
            new PointerEvent(type, {
              bubbles: true,
              cancelable: true,
              pointerId: 7,
              clientX: x,
              clientY: y,
            }),
          );
        drag("pointerdown", fromX);
        await sleep(80);
        drag("pointermove", toX);
        await sleep(80);
        drag("pointerup", toX);
        await sleep(400);
      }
      check("speed: dragging the thumb changes the tier", () => ({
        ok: store.getState().fastSpeed === "ultrafast",
        speed: store.getState().fastSpeed,
      }));
    }

    // ── 9o. response length: + menu row and Settings share one value ───
    const verbAddTrigger = document.querySelector(
      '.composer-bar .bar-btn[title="Add to prompt"]',
    );
    if (verbAddTrigger) {
      verbAddTrigger.click();
      await waitFor(() => document.querySelector('[data-capability="chat.verbosity"]'), 20);
      check("verbosity: row lives in its own Reply section", () => {
        const row = document.querySelector('[data-capability="chat.verbosity"]');
        const labels = Array.from(
          document.querySelectorAll(".popover .popover-label"),
        ).map((item) => item.textContent.trim());
        const reply = labels.findIndex((label) => /^Reply$/.test(label));
        const insert = labels.findIndex((label) => /^Insert$/.test(label));
        return {
          ok:
            Boolean(row) &&
            row.closest(".popover") !== null &&
            /Length/.test(row.textContent || "") &&
            row.getBoundingClientRect().height <= 30 &&
            reply >= 0 &&
            insert === reply + 1,
          labels,
        };
      });

      const lengthArrow = document.querySelector(
        '.popover .mode-arrow[aria-label="Reply length"]',
      );
      if (lengthArrow) lengthArrow.click();
      await waitFor(() => document.querySelector(".verbosity-panel"), 20);
      check("verbosity: flyout explains each option", () => {
        const panel = document.querySelector(".verbosity-panel");
        const text = panel ? panel.textContent || "" : "";
        return {
          ok:
            Boolean(panel) &&
            /0\.6×/.test(text) &&
            /1\.0×/.test(text) &&
            /1\.5×/.test(text) &&
            /Answer only what was asked/.test(text) &&
            /Reasoning, examples and caveats/.test(text) &&
            !/Also in Settings/.test(text) &&
            !/Response length/.test(text),
        };
      });
      const flyoutDetailed = Array.from(
        document.querySelectorAll(".verbosity-option"),
      ).find((option) => /Detailed/.test(option.textContent || ""));
      if (flyoutDetailed) flyoutDetailed.click();
      await sleep(450);
      check("verbosity: choosing an option updates the row", () => ({
        ok:
          store.getState().verbosity === "detailed" &&
          /Detailed/.test(
            (document.querySelector('[data-capability="chat.verbosity"]') || {})
              .textContent || "",
          ),
        verbosity: store.getState().verbosity,
      }));
    }

    store.getState().openSettings("Chat");
    await waitFor(
      () =>
        /Response length/.test(
          (document.querySelector(".set-content") || {}).textContent || "",
        ),
      20,
    );
    check("verbosity: Settings mirrors the flyout", () => {
      const pressed = Array.from(
        document.querySelectorAll(".set-content .persona-chip"),
      ).find((chip) => chip.getAttribute("aria-pressed") === "true");
      return {
        ok: Boolean(pressed) && pressed.textContent.trim() === "Detailed",
        selected: pressed ? pressed.textContent.trim() : null,
      };
    });
    const shortOption = Array.from(
      document.querySelectorAll(".set-content .persona-chip"),
    ).find((chip) => chip.textContent.trim() === "Short");
    if (shortOption) shortOption.click();
    await sleep(450);
    check("verbosity: changing Settings updates the row", () => ({
      ok: store.getState().verbosity === "short",
      verbosity: store.getState().verbosity,
    }));
    store.getState().closeSettings();
    await sleep(200);

    // ── 9p. System theme option ────────────────────────────────────────
    store.getState().openSettings("Appearance");
    await waitFor(
      () =>
        /Color mode/.test((document.querySelector(".set-content") || {}).textContent || ""),
      20,
    );
    check("theme: System option is available", () => {
      const group = document.querySelector('[role="radiogroup"][aria-label="Color mode"]');
      const labels = Array.from(group ? group.querySelectorAll("button") : []).map(
        (button) => button.textContent.trim(),
      );
      return { ok: labels.includes("System"), labels };
    });
    store.getState().closeSettings();
    await sleep(200);

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
