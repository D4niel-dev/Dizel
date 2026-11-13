import time
import json
import os
from .config_manager import SETTINGS_FILE

class UsageManager:
    """Manages AI compute usage, complexity scoring, and 12-hour reset windows."""
    
    RESET_WINDOW = 12 * 3600  # 12 hours in seconds

    COMPACTION_THRESHOLD = 84        # % of capacity that triggers auto-compaction
    MAX_COMPACTIONS = 5              # compactions allowed per 12h window

    def __init__(self):
        self._current_usage = 0.0
        self._last_reset = time.time()
        self._max_capacity = 1024.0  # Default base capacity
        self._compaction_count  = 0

        self.load_state()
        self.check_reset()

    def set_capacity(self, capacity: float):
        """Dynamic capacity based on model checkpoint (e.g. 1024 or 4096).

        Note the unit: capacity is expressed in the same "tokens" the cost model
        charges. At WORDS_PER_TOKEN = 10, a 2,048 capacity is ~20,480 words —
        roughly 400 ordinary messages — with compaction at ~17,000 words. Generous
        but reachable. Raise WORDS_PER_TOKEN to stretch it, lower it to tighten.
        """
        if capacity > 0:
            self._max_capacity = float(capacity)

    def load_state(self):
        try:
            # We store usage in the same dir as settings for simplicity
            path = SETTINGS_FILE.replace("settings.json", "usage_state.json")
            if os.path.exists(path):
                with open(path, "r") as f:
                    data = json.load(f)
                    self._current_usage = data.get("usage", 0.0)
                    self._last_reset = data.get("last_reset", time.time())
                    self._compaction_count  = data.get("compaction_count", 0)
        except:
            pass

    def save_state(self):
        try:
            path = SETTINGS_FILE.replace("settings.json", "usage_state.json")
            with open(path, "w") as f:
                json.dump({
                    "usage": self._current_usage,
                    "last_reset": self._last_reset,
                    "compaction_count": self._compaction_count,
                }, f)
        except:
            pass

    def check_reset(self):
        now = time.time()
        if now - self._last_reset > self.RESET_WINDOW:
            self._current_usage = 0.0
            self._compaction_count = 0
            self._last_reset = now
            self.save_state()

    def add_usage(self, amount: float):
        self.check_reset()
        self._current_usage = min(self._current_usage + amount, self._max_capacity)
        self.save_state()

    @property
    def usage(self) -> float:
        self.check_reset()
        return self._current_usage

    @property
    def percentage(self) -> int:
        return min(100, int((self._current_usage / max(self._max_capacity, 1)) * 100))

    @property
    def max_capacity(self) -> float:
        return self._max_capacity

    @property
    def is_exhausted(self) -> bool:
        """True if current usage meets or exceeds max capacity."""
        return self._current_usage >= self._max_capacity

    @property
    def needs_compaction(self) -> bool:
        """True when usage has reached the 84% compaction threshold."""
        return self.percentage >= self.COMPACTION_THRESHOLD

    @property
    def compaction_count(self) -> int:
        return self._compaction_count

    @property
    def compactions_remaining(self) -> int:
        return max(0, self.MAX_COMPACTIONS - self._compaction_count)

    def can_compact(self) -> bool:
        """Compaction allowed until 5 have been used in this window."""
        return self._compaction_count < self.MAX_COMPACTIONS

    # ── Cost model ──────────────────────────────────────────────────────
    # The window is a *policy* number, not a physical one — it exists to stop a
    # session running forever, not to predict the tokenizer. So it uses a fixed,
    # generous rate instead of real counts:
    #
    #     10 words      ->   1 token       (10 words per token)
    #     10 tool calls ->   5 tokens      (0.5 per call)
    #
    # The rate is the dial, not the capacity: capacity keeps meaning "the model's
    # context size", and only the exchange rate decides how fast it drains. At 10
    # words/token a 2,048 window holds ~20,000 words — generous, but reachable.
    #
    # The context meter is a different thing and keeps real counting: that one is
    # a hard limit the model enforces, and a generous number there would only hide
    # truncation rather than prevent it.
    WORDS_PER_TOKEN = 10.0
    TOKENS_PER_TOOL_CALL = 0.5

    def charge_text(self, text: str) -> float:
        """Charge for a piece of text at the fixed words-per-token rate."""
        words = len((text or "").split())
        if words <= 0:
            return 0.0
        cost = words / self.WORDS_PER_TOKEN
        self.add_usage(cost)
        return cost

    def charge_tool_calls(self, count: int = 1) -> float:
        """Charge for tool calls — flat, so a chatty model cannot drain the window."""
        if count <= 0:
            return 0.0
        cost = count * self.TOKENS_PER_TOOL_CALL
        self.add_usage(cost)
        return cost

    # ── Debug levers ────────────────────────────────────────────────────
    # Named entry points so tests and debug commands do not reach into
    # `_current_usage` directly. Each one persists, exactly like real usage.
    def force_usage(self, amount: float) -> None:
        self._current_usage = max(0.0, min(float(amount), self._max_capacity))
        self.save_state()

    def force_percent(self, percent: float) -> None:
        self.force_usage(self._max_capacity * max(0.0, min(100.0, percent)) / 100.0)

    def force_reset(self) -> None:
        self._current_usage = 0.0
        self._compaction_count = 0
        self._last_reset = time.time()
        self.save_state()

    def expire_window(self) -> None:
        """Pretend the 12-hour window has elapsed, so reset logic can be seen."""
        self._last_reset = time.time() - self.RESET_WINDOW - 1
        self.save_state()

    def run_compaction(self) -> None:
        """Count the compaction and reset usage back to 0%."""
        self._compaction_count += 1
        self._current_usage = 0.0
        self.check_reset()
        self.save_state()

    def time_until_reset(self) -> str:
        """Return human-readable time until the 12-hour usage window resets."""
        remaining = self.RESET_WINDOW - (time.time() - self._last_reset)
        if remaining <= 0:
            return "any moment"
        hours = int(remaining // 3600)
        minutes = int((remaining % 3600) // 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"
