// Injects a text attachment into the composer, opens its preview, and checks
// that a binary file is refused rather than decoded into the prompt.
(async () => {
  const report = {};

  const binary = await window.dizel.readTextFile(
    "C:/Users/KHAC DUY/Desktop/Dizel/inference/dizel_gui/assets/app/Dizel.png",
  );
  report.binaryRejected = !binary.ok;
  report.binaryReason = binary.error ?? null;

  const text = await window.dizel.readTextFile(
    "C:/Users/KHAC DUY/Desktop/Dizel/inference/dizel_gui/logic/config_manager.py",
  );
  report.textOk = text.ok;
  report.textChars = text.ok ? text.content.length : 0;

  window.__dizel.setState({
    attachments: [
      {
        id: "a1",
        name: "config_manager.py",
        path: "C:/Users/KHAC DUY/Desktop/Dizel/inference/dizel_gui/logic/config_manager.py",
        size: text.ok ? text.size : 48210,
        content: text.ok ? text.content.slice(0, 6000) : "",
        truncated: text.ok ? text.truncated : false,
      },
    ],
  });

  await new Promise((r) => setTimeout(r, 400));
  document.querySelector(".chip").click();
  await new Promise((r) => setTimeout(r, 300));

  report.chipCount = document.querySelectorAll(".chip").length;
  report.previewOpen = Boolean(document.querySelector(".attach-preview"));

  return JSON.stringify(report);
})();
