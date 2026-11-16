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
    await waitFor(() => store.getState().approval === null, 20);
    check("trust: always-this-session resolves the synthetic gate", () => ({
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
    check("trust: settings offers the persistent switch", () => ({ ok: Boolean(trustSwitch) }));
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
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "k", ctrlKey: true, bubbles: true }));
    const palInput = await waitFor(() => document.querySelector(".command-palette input"), 20);
    check("palette: opens on Ctrl+K", () => ({
      ok: Boolean(palInput) && store.getState().paletteOpen === true,
      found: Boolean(palInput),
      open: store.getState().paletteOpen,
    }));
    if (palInput) {
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
