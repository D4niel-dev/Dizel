// Creates a session containing a <think> block through the app's own API and
// opens it, so the reasoning renderer can be checked without a reasoning model.
// The session id is returned so it can be deleted afterwards.
(async () => {
  const info = await window.dizel.getSidecar();
  const base = `http://127.0.0.1:${info.port}/api`;
  const headers = {
    "Content-Type": "application/json",
    "X-Dizel-Token": info.token,
  };

  const payload = {
    session_id: null,
    title: "Think block check",
    messages: [
      { role: "user", content: "Why is the sky blue?" },
      {
        role: "assistant",
        content:
          "<think>The user asks about Rayleigh scattering. Keep it to two sentences, no formulas.</think>" +
          "The sky looks blue because air molecules scatter short wavelengths of light far more than long ones. " +
          "That scattered blue reaches your eye from every direction, so the whole sky glows.",
      },
    ],
  };

  const created = await fetch(`${base}/session/`, {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
  }).then((response) => response.json());

  await window.__dizel.getState().loadSessions();
  await window.__dizel.getState().openSession(created.session_id);

  await new Promise((resolve) => setTimeout(resolve, 500));

  return JSON.stringify({
    created: created.session_id,
    thinkBlocks: document.querySelectorAll(".think").length,
  });
})();
