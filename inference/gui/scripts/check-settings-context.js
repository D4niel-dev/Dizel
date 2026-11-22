// Settings → Chat → Project context, in both states.
//
// The live sidecar answers for this checkout (which has no rule files), so the
// populated layout is produced by stubbing only the project/memory request.
(async () => {
  const store = window.__dizel;
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  for (let i = 0; i < 40; i += 1) {
    if (store.getState().booted) break;
    await wait(250);
  }

  store.setState({ theme: "dark" });
  document.documentElement.dataset.theme = "dark";

  const realFetch = window.fetch.bind(window);
  let stub = null;
  window.fetch = (input, init) => {
    const url = typeof input === "string" ? input : input.url;
    if (stub && url.includes("/project/memory")) {
      return Promise.resolve(
        new Response(JSON.stringify(stub), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      );
    }
    return realFetch(input, init);
  };

  // The section grows when its request lands, so pin the row itself rather
  // than a scroll offset measured before that happened.
  const scrollToContext = () => {
    const label = Array.from(document.querySelectorAll(".set-row-text > span")).find(
      (node) => node.textContent === "Project rules",
    );
    label?.closest(".set-row")?.scrollIntoView({ block: "center" });
  };

  const rowText = () => {
    const rows = Array.from(document.querySelectorAll(".set-row"));
    return rows
      .map((row) => ({
        label: row.querySelector(".set-row-text > span")?.textContent ?? "",
        hint: row.querySelector(".set-row-text small")?.textContent ?? "",
        value: row.querySelector(".set-row-control")?.textContent?.trim() ?? "",
        spec: row.classList.contains("is-spec"),
        hintLines: (() => {
          const small = row.querySelector(".set-row-text small");
          if (!small) return 0;
          return Math.round(small.getBoundingClientRect().height / parseFloat(getComputedStyle(small).lineHeight || "16"));
        })(),
      }))
      .filter((row) => row.label === "Project rules" || row.label === "Project memory");
  };

  // The dialog mounts once at startup, so its load effect only reruns on a
  // genuine close/reopen. Close first, or the stub is never consulted.
  const open = async () => {
    store.getState().closeSettings();
    await wait(250);
    store.getState().openSettings("Chat");
    await wait(1600);
  };

  // ── Populated ────────────────────────────────────────────────────────
  stub = {
    project: "C:/work/demo",
    entries: [
      { id: "m1", text: "Use the measurement bench tokens.", created: "", source: "user" },
      { id: "m2", text: "Never commit secrets.", created: "", source: "user" },
    ],
    rules: [
      { name: "DIZEL.md", path: "DIZEL.md", chars: 2418 },
      { name: "AGENTS.md", path: "AGENTS.md", chars: 902 },
    ],
    rules_summary: "DIZEL.md — 2,418 chars\nAGENTS.md — 902 chars",
    rules_truncated: false,
  };
  await open();
  scrollToContext();
  await wait(200);
  const populated = rowText();

  // ── Empty (this checkout) ────────────────────────────────────────────
  stub = null;
  await open();
  scrollToContext();
  await wait(400);
  scrollToContext();
  await wait(200);
  const empty = rowText();

  return JSON.stringify({
    populated,
    empty,
    settingsOpen: store.getState().settingsOpen,
    rows: document.querySelectorAll(".set-row").length,
    sections: Array.from(document.querySelectorAll(".set-section-head, .set-section > h3, section h3")).length,
    dialog: Boolean(document.querySelector(".modal, .settings-modal, [role=dialog]")),
    labels: Array.from(document.querySelectorAll(".set-row-text > span")).slice(0, 12).map((n) => n.textContent),
  });
})();
