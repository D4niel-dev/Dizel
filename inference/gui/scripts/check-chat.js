// Drives a real chat turn through the store, then reports what landed.
// Used by the headless screenshot check (DIZEL_SHOT_JS_FILE).
(async () => {
  const store = window.__dizel;

  for (let i = 0; i < 90; i += 1) {
    const state = store.getState();
    if (state.status && state.status.is_ready) break;
    await new Promise((r) => setTimeout(r, 500));
  }

  const before = store.getState();
  if (!before.status || !before.status.is_ready) {
    return JSON.stringify({ ok: false, reason: "model never became ready", load: before.status && before.status.load });
  }

  await store.getState().send("Explain what a variable is in one sentence.");

  await new Promise((r) => setTimeout(r, 800));

  const after = store.getState();
  const last = after.messages[after.messages.length - 1];
  return JSON.stringify({
    ok: true,
    count: after.messages.length,
    streaming: after.streaming,
    error: after.error,
    chars: last ? last.content.length : 0,
    text: last ? last.content.slice(0, 160) : "",
    bubbles: document.querySelectorAll(".msg").length,
    markdownBlocks: document.querySelectorAll(".md").length,
  });
})();
