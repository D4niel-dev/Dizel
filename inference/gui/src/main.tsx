import { createRoot } from "react-dom/client";

import App from "./App";
import { useChat } from "./store/chatStore";
import "./styles/tokens.css";
import "./styles/app.css";

const container = document.getElementById("root");
if (!container) throw new Error("Root container missing");

// Debug handle, only mounted when the shell explicitly asks for it
// (?debug=1, set by DIZEL_DEBUG=1). Used for automated UI checks.
if (new URLSearchParams(window.location.search).has("debug")) {
  (window as unknown as { __dizel: unknown }).__dizel = useChat;
}

createRoot(container).render(<App />);
