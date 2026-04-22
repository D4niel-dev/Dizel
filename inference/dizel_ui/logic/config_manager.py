"""
dizel_ui/logic/config_manager.py
────────────────────────────────
Handles saving and loading of application settings to disk.
Settings are persisted as JSON in the same directory as chat history.
"""

import json
import os
from typing import Any, Dict

# Settings will live inside inference/dizel_ui/.dizel/settings.json
_HERE = os.path.dirname(os.path.abspath(__file__))
_UI_DIR = os.path.dirname(_HERE)
_DATA_DIR = os.path.join(_UI_DIR, ".dizel")

SETTINGS_FILE = os.path.join(_DATA_DIR, "settings.json")


class ConfigManager:
    """Manages application-wide persistent settings."""
    
    _DEFAULTS = {
        "checkpoint": "",
        "device": "cpu",
        "system_prompt": (
            "You are Dizel, a highly capable, intelligent, and helpful AI assistant. "
            "You answer thoughtfully, concisely, and accurately. "
            "You use formatting like markdown to organize your thoughts and provide clear, structured text."
        ),
        "sampling": {
            "temperature": 0.4,
            "top_k": 90,
            "top_p": 0.92,
            "repetition_penalty": 1.15,
            "max_new_tokens": 200,
        },
        "appearance": {
            "theme": "dark"
        },
        "user_profile": {
            "username": "User",
            "avatar": ""
        },
        "token_budget": {
            "chat_budget": 150,
            "coding_budget": 350,
            "complex_budget": 500,
            "factual_budget": 100,
            "tool_budget": 300,
            "max_context_tokens": 3500,
            "verbosity": "normal",
            "hard_output_limit": 600,
        },
        "tutorial": {
            "completed": False,
            "skipped": False,
            "rating": 0
        }
    }

    @classmethod
    def load(cls) -> Dict[str, Any]:
        """Load settings from disk, falling back to defaults if missing/corrupt."""
        if not os.path.exists(SETTINGS_FILE):
            return cls._DEFAULTS.copy()

        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # Merge with defaults to ensure all keys exist even if older version
                merged = cls._DEFAULTS.copy()
                for key, val in data.items():
                    if isinstance(val, dict) and key in merged and isinstance(merged[key], dict):
                        merged[key].update(val)
                    else:
                        merged[key] = val
                return merged
        except Exception as e:
            print(f"Warning: Failed to load settings.json ({e}). Using defaults.")
            return cls._DEFAULTS.copy()

    @classmethod
    def save(cls, settings: Dict[str, Any]) -> None:
        """Save settings dictionary to disk."""
        os.makedirs(_DATA_DIR, exist_ok=True)
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            print(f"Error: Failed to save settings.json ({e})")
