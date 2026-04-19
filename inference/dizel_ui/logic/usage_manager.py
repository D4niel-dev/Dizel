import time
import json
import os
from dizel_ui.logic.config_manager import SETTINGS_FILE

class UsageManager:
    """Manages AI compute usage, complexity scoring, and 12-hour reset windows."""
    
    RESET_WINDOW = 12 * 3600  # 12 hours in seconds
    
    def __init__(self):
        self._current_usage = 0.0
        self._last_reset = time.time()
        self._max_capacity = 1024.0  # Default base capacity
        
        self.load_state()
        self.check_reset()

    def set_capacity(self, capacity: float):
        """Dynamic capacity based on model checkpoint (e.g. 1024 or 4096)."""
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
        except:
            pass

    def save_state(self):
        try:
            path = SETTINGS_FILE.replace("settings.json", "usage_state.json")
            with open(path, "w") as f:
                json.dump({
                    "usage": self._current_usage,
                    "last_reset": self._last_reset
                }, f)
        except:
            pass

    def check_reset(self):
        now = time.time()
        if now - self._last_reset > self.RESET_WINDOW:
            self._current_usage = 0.0
            self._last_reset = now
            self.save_state()

    def add_usage(self, amount: float):
        self.check_reset()
        self._current_usage += amount
        self.save_state()

    @property
    def usage(self) -> float:
        self.check_reset()
        return self._current_usage

    @property
    def percentage(self) -> int:
        return int((self._current_usage / self._max_capacity) * 100)

    @property
    def max_capacity(self) -> float:
        return self._max_capacity
