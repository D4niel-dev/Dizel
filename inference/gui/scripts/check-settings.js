// Toggles "UI animations" off in the Appearance tab and saves, then reports
// whether the preference reached the DOM and the store.
(async () => {
  const store = window.__dizel;
  const report = {};
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  const waitFor = async (fn, label, timeout = 8000) => {
    const started = Date.now();
    while (Date.now() - started < timeout) {
      const value = fn();
      if (value) return value;
      await sleep(150);
    }
    throw new Error(`timed out waiting for ${label}`);
  };

  store.getState().openSettings();
  await waitFor(() => document.querySelectorAll(".set-nav-item").length, "nav");

  const navItems = document.querySelectorAll(".set-nav-item");
  report.navCount = navItems.length;
  navItems[2].click(); // Appearance

  // The tab renders its rows only after the config request resolves.
  const toggles = await waitFor(() => {
    const found = document.querySelectorAll(".set-row.as-button");
    return found.length >= 5 ? found : null;
  }, "appearance rows");

  report.toggleCount = toggles.length;
  report.motionBefore = document.documentElement.dataset.motion;

  toggles[3].click(); // UI animations
  await sleep(200);

  document.querySelector(".set-foot .btn.primary").click();
  await sleep(1600);

  report.dialogClosed = !store.getState().settingsOpen;
  report.settingsError = store.getState().settingsError;
  report.animationsPref = store.getState().prefs.animations;
  report.motionAfter = document.documentElement.dataset.motion;

  return JSON.stringify(report);
})();
