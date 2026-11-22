// Headless check for the newer shell features, driven through the real store.
//
//   DIZEL_PROD=1 DIZEL_DEBUG=1 DIZEL_SHOT=out.png DIZEL_SHOT_DELAY=9000 \
//     DIZEL_SHOT_JS_FILE=scripts/check-new-ui.js npx electron . --disable-gpu
//
// Returns JSON so the harness logs a verdict per feature rather than a picture
// someone has to squint at.
(async () => {
  const store = window.__dizel;
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  const results = {};

  for (let i = 0; i < 40; i += 1) {
    if (store.getState().booted) break;
    await wait(250);
  }

  // ── Streaming markdown still renders block by block ──────────────────
  const streamNow = Date.now();
  const block =
    "A paragraph with **bold** and `code`.\n\n- one\n- two\n\n| A | B |\n| :-- | --: |\n| 1 | 2 |\n\n```python\nprint(1)\n```\n\n";
  store.setState({
    messages: [
      { id: "u2", role: "user", content: "Render.", createdAt: streamNow },
      {
        id: "a2",
        role: "assistant",
        content: block.repeat(3) + "Still writing",
        createdAt: streamNow,
        streaming: true,
      },
    ],
  });
  await wait(600);
  const md = document.querySelector(".md");
  results.streamingMarkdown = {
    roots: document.querySelectorAll(".md").length,
    lists: md ? md.querySelectorAll("ul").length : 0,
    tables: md ? md.querySelectorAll("table").length : 0,
    code: md ? md.querySelectorAll(".code-block").length : 0,
    rawFencesVisible: document.body.innerText.includes("```"),
  };

  const now = Date.now();
  const diff =
    "--- a/src/components/Welcome.tsx\n+++ b/src/components/Welcome.tsx\n@@ -1,4 +1,6 @@\n-export function Welcome() {\n+export function Welcome({ model }: { model: string }) {\n+  return <h1>Ask {model} anything</h1>;\n }\n";

  store.setState({
    booted: true,
    theme: "dark",
    verbosity: "detailed",
    artifactsOpen: true,
    messages: [
      { id: "u1", role: "user", content: "Show me the diff.", createdAt: now - 60000 },
      {
        id: "a1",
        role: "assistant",
        createdAt: now - 30000,
        content:
          "Here is the change.\n\n```tsx\nexport function Welcome({ model }: { model: string }) {\n  return <h1>Ask {model} anything</h1>;\n}\n```\n",
        tools: [
          {
            id: 1,
            name: "edit_file",
            status: "done",
            path: "src/components/Welcome.tsx",
            action: "edit",
            additions: 2,
            deletions: 1,
            detail: "src/components/Welcome.tsx",
            diff,
          },
          {
            id: 2,
            name: "read_file",
            status: "done",
            path: "src/App.tsx",
            detail: "src/App.tsx:1-12",
            result: "src/App.tsx — lines 1-12 of 48\nimport { useState } from 'react';\n",
          },
        ],
      },
    ],
    sessions: [
      {
        id: "s1",
        title: "Empty state pass",
        created: "2026-11-18T09:00:00",
        preview: "",
        pinned: true,
        archived: false,
        tags: ["design", "gui", "polish"],
      },
    ],
  });
  document.documentElement.dataset.theme = "dark";
  await wait(350);

  // ── Artifacts rail ────────────────────────────────────────────────────
  store.getState().openArtifacts();
  await wait(250);
  results.railOpens = Boolean(document.querySelector(".art-panel"));
  results.fileRows = document.querySelectorAll(".art-row").length;
  // The newest row is the read; the diff belongs to the edited file.
  const edited = Array.from(document.querySelectorAll(".art-row")).find((row) =>
    row.querySelector(".art-stats"),
  );
  edited?.querySelector(".art-row-head")?.click();
  await wait(200);
  results.diffVisible = Boolean(document.querySelector(".art-row-body .tool-diff"));

  const codeTab = Array.from(document.querySelectorAll(".art-tab")).find((tab) =>
    tab.textContent.startsWith("Code"),
  );
  codeTab?.click();
  await wait(150);
  results.codeCards = document.querySelectorAll(".art-code").length;
  document.querySelector(".art-code-toggle")?.click();
  await wait(150);
  results.codeHighlighted = Boolean(document.querySelector(".art-code-body .hljs"));

  // ── Reply length ──────────────────────────────────────────────────────
  const plus = document.querySelector(".composer-bar .bar-btn");
  results.plusArmed = plus ? plus.className.includes("armed") : false;
  plus?.click();
  await wait(200);
  results.lengthSection = Array.from(document.querySelectorAll(".popover-label")).some(
    (node) => node.textContent.includes("Response length"),
  );
  results.lengthOptions = Array.from(document.querySelectorAll(".popover-item")).filter(
    (node) => ["Short", "Normal", "Detailed"].some((word) => node.textContent.includes(word)),
  ).length;
  results.lengthInUse = Array.from(document.querySelectorAll(".popover-hint")).some(
    (node) => node.textContent === "in use",
  );
  plus?.click();

  // ── Tags ─────────────────────────────────────────────────────────────
  results.rowTags = Array.from(document.querySelectorAll(".sb-row-tag")).map(
    (node) => node.textContent,
  );
  const more = document.querySelector(".sb-row .row-act");
  more?.click();
  await wait(150);
  results.menuItems = Array.from(document.querySelectorAll(".row-menu-item")).map(
    (node) => node.textContent,
  );
  const tags = Array.from(document.querySelectorAll(".row-menu-item")).find((node) =>
    node.textContent.toLowerCase().includes("tag"),
  );
  tags?.click();
  await wait(200);
  const tagInput = document.querySelector(".sb-row.editing input");
  results.tagEditor = Boolean(tagInput);
  results.tagFocused = document.activeElement === tagInput;
  if (tagInput) {
    tagInput.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  }

  // ── Rail exclusivity ─────────────────────────────────────────────────
  store.getState().setPreview({
    id: "f1",
    name: "notes.md",
    path: "notes.md",
    size: 12,
    content: "# Notes\n",
    truncated: false,
  });
  await wait(150);
  results.previewClosedRail = store.getState().artifactsOpen === false;
  store.getState().openArtifacts();
  await wait(150);
  results.railClosedPreview = store.getState().preview === null;

  // ── Status bar carries the length ────────────────────────────────────
  const bar = document.querySelector(".statusbar");
  results.statusBar = bar ? bar.textContent.replace(/\s+/g, " ").trim() : null;

  return JSON.stringify(results);
})();
