from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Static, Button
from textual.containers import Vertical
from textual import events, on

AVAILABLE_TOOLS = {
    "web_search": "Search the web for relevant information",
    "web_scraper": "Extract raw text content from URLs",
    "file_parse": "Parse and extract context from attached files",
    "deep_think": "Extended reasoning with higher token budget",
    "thinking": "Lightweight reasoning before responding",
    "python_interpreter": "Execute Python scripts natively",
}

class ToolsMenuModal(ModalScreen[None]):
    """A modal overlay for toggling active AI tools."""

    CSS = """
    ToolsMenuModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.7);
    }

    #tools-container {
        width: 60;
        height: auto;
        background: #18181B;
        border: ascii #3B82F6;
        padding: 1 2;
    }

    #tools-header {
        height: 1;
        text-align: center;
        color: #60A5FA;
        text-style: bold;
        margin-bottom: 1;
    }

    .tool-row {
        height: 4;
        margin-bottom: 1;
    }
    
    Button {
        width: 100%;
        height: 3;
        border: none;
        padding: 0 1;
    }

    .tool-desc {
        color: #A1A1AA;
        padding-left: 2;
        margin-top: 1;
    }

    #tools-footer {
        height: 1;
        text-align: right;
        color: #71717A;
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="tools-container"):
            yield Static("AI Tools Configuration", id="tools-header")
            
            # Ensure the app has a tools state dict
            if not hasattr(self.app, "_tool_states"):
                self.app._tool_states = {name: True for name in AVAILABLE_TOOLS}
                
            for name, desc in AVAILABLE_TOOLS.items():
                with Vertical(classes="tool-row"):
                    is_active = self.app._tool_states.get(name, True)
                    label = "[X]" if is_active else "[ ]"
                    yield Button(f"{label} {name.replace('_', ' ').title()}", id=f"btn_{name}", variant="primary" if is_active else "default")
                    yield Static(desc, classes="tool-desc")
                    
            yield Static("[esc] close menu", id="tools-footer")

    @on(Button.Pressed)
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if not event.button.id or not event.button.id.startswith("btn_"):
            return
        tool_name = event.button.id.replace("btn_", "")
        current_state = self.app._tool_states.get(tool_name, True)
        new_state = not current_state
        self.app._tool_states[tool_name] = new_state
        
        # Update button visual
        label = "[X]" if new_state else "[ ]"
        event.button.label = f"{label} {tool_name.replace('_', ' ').title()}"
        event.button.variant = "primary" if new_state else "default"

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            self.dismiss(None)
