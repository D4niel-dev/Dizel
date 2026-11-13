// Verifies the new controls: the file bridge, mode switching, and sending.
// Driven by the headless screenshot hook (DIZEL_SHOT_JS_FILE).
(async () => {
  const store = window.__dizel;
  const report = {};

  // 1. File bridge (same path the + button uses, minus the native dialog).
  const read = await window.dizel.readTextFile(
    "C:/Users/KHAC DUY/Desktop/Dizel/README.md",
  );
  report.readFile = read.ok ? `ok, ${String(read.content || "").length} chars` : read.error;

  // 2. Wait for the model.
  for (let i = 0; i < 120; i += 1) {
    const state = store.getState();
    if (state.status && state.status.is_ready) break;
    await new Promise((r) => setTimeout(r, 500));
  }
  report.ready = Boolean(store.getState().status && store.getState().status.is_ready);
  if (!report.ready) return JSON.stringify(report);

  // 3. Mode switch should apply a profile without raising.
  await store.getState().setMode("Planning");
  report.mode = store.getState().mode;
  report.modeError = store.getState().error;

  // 4. Send with the new prompt path.
  await store.getState().send("Say hello in exactly three words.");
  await new Promise((r) => setTimeout(r, 600));

  const after = store.getState();
  const last = after.messages[after.messages.length - 1];
  report.count = after.messages.length;
  report.sendError = after.error;
  report.replyChars = last ? last.content.length : 0;
  report.reply = last ? last.content.slice(0, 90) : "";
  report.bubbles = document.querySelectorAll(".msg").length;

  return JSON.stringify(report);
})();
