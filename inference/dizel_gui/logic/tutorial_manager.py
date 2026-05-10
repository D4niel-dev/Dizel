from dataclasses import dataclass
from typing import Dict, List, Optional
from .config_manager import ConfigManager

@dataclass
class TutorialStep:
    id: str
    title: str
    body: str
    target_widget_id: str
    spotlight: bool
    action_id: Optional[str] = None
    require_action: bool = False

class TutorialState:
    def __init__(self, cfg: dict):
        self.completed: bool = cfg.get("completed", False)
        self.skipped: bool = cfg.get("skipped", False)
        self.rating: int = cfg.get("rating", 0)
        self.current_step_index: int = 0
        self.checkpoint_exists: bool = False

class TutorialManager:
    """Manages the logic, state, and step definitions for the first-run tutorial."""
    _STEPS = [
        TutorialStep(
            id="welcome",
            title="Welcome to Dizel! 👋",
            body="Dizel is your personal AI assistant. Chat, brainstorm, analyze files, search the web — all in one place. Let's take a quick tour!",
            target_widget_id="welcome",
            spotlight=False
        ),
        TutorialStep(
            id="sidebar",
            title="Your Workspace",
            body="This is your workspace. Create new chats, browse saved conversations, and organize your work into workspaces.",
            target_widget_id="sidebar",
            spotlight=True
        ),
        TutorialStep(
            id="config",
            title="Configuration & Model",
            body="First, let's look at Configuration. That gear icon opens a beautiful control panel for models.",
            target_widget_id="config",
            spotlight=True
            # No action_id here, just pointing at the gear icon
        ),
        TutorialStep(
            id="api_router",
            title="Meet the API Router",
            body="Dizel natively supports cloud models right alongside local checkpoints. Choose Groq, OpenAI, Anthropic, and more right here.",
            target_widget_id="api_grid",
            spotlight=True,
            action_id="open_modal_settings" # Opens settings immediately when arriving at this step
        ),
        TutorialStep(
            id="tools",
            title="Powerful Tools",
            body="Click the + button to access powerful tools: Web Search, Deep Think, file parsing, and more. Each tool changes how Dizel responds.",
            target_widget_id="tools",
            spotlight=True,
            action_id="close_modal_settings" # Closes the modal returning to the main screen
        ),
        TutorialStep(
            id="modes",
            title="Model & Mode",
            body="Switch between model variants (Lite/Pro) and modes (Fast/Planning). Each combination has a different personality and response style.",
            target_widget_id="modes",
            spotlight=True
        ),
        TutorialStep(
            id="first_message",
            title="Send Your First Message",
            body="Type something in the text box below and press Enter! The tutorial will wait for you to do this.",
            target_widget_id="input",
            spotlight=True,
            require_action=True
        ),
        TutorialStep(
            id="finish",
            title="You're All Set! ⭐",
            body="That's it! You're ready to go. How was this tutorial?",
            target_widget_id="finish",
            spotlight=False
        )
    ]

    def __init__(self):
        self.cfg = ConfigManager.load()
        tut_cfg = self.cfg.get("tutorial", {})
        self.state = TutorialState(tut_cfg)

    def should_show(self) -> bool:
        return not self.state.completed and not self.state.skipped

    def get_current_step(self) -> TutorialStep:
        return self._STEPS[self.state.current_step_index]
        
    def get_total_steps(self) -> int:
        return len(self._STEPS)

    def next_step(self) -> bool:
        if self.state.current_step_index < len(self._STEPS) - 1:
            self.state.current_step_index += 1
            return True
        return False

    def prev_step(self) -> bool:
        if self.state.current_step_index > 0:
            self.state.current_step_index -= 1
            return True
        return False

    def skip(self):
        self.state.skipped = True
        self._save_state()

    def complete(self, rating: int):
        self.state.rating = rating
        self.state.completed = True
        self._save_state()

    def reset(self):
        self.state.completed = False
        self.state.skipped = False
        self.state.rating = 0
        self.state.current_step_index = 0
        self._save_state()

    def _save_state(self):
        # Update config and persist
        self.cfg["tutorial"] = {
            "completed": self.state.completed,
            "skipped": self.state.skipped,
            "rating": self.state.rating
        }
        ConfigManager.save(self.cfg)
