import json
import os
import time

from . import config_manager


class UsageManager:
    """Manages AI compute usage per provider, with 12-hour reset windows.

    Every provider (local checkpoints included) keeps its own bucket: its own
    spent amount, window start, and compaction count. Switching providers
    switches the meter; switching back restores that provider's window.
    """

    RESET_WINDOW = 12 * 3600  # 12 hours in seconds

    COMPACTION_THRESHOLD = 84        # % of capacity that triggers auto-compaction
    MAX_COMPACTIONS = 5              # compactions allowed per 12h window
    STATE_VERSION = 2
    DEFAULT_CAPACITY = 1024.0

    def __init__(self):
        self._active = "local"
        self._providers: dict[str, dict] = {}
        self._capacities: dict[str, float] = {}

        self.load_state()
        self._bucket()
        self.check_reset()

    # ── Provider buckets ────────────────────────────────────────────────
    @staticmethod
    def _slug(provider: str | None) -> str:
        return str(provider or "").strip().lower() or "local"

    def _bucket(self, provider: str | None = None) -> dict:
        slug = self._slug(provider) if provider is not None else self._active
        return self._providers.setdefault(
            slug,
            {"usage": 0.0, "last_reset": time.time(), "compaction_count": 0},
        )

    @property
    def active_provider(self) -> str:
        return self._active

    def activate(self, provider: str, capacity: float | None = None) -> None:
        """Switch the active bucket (and optionally its capacity)."""
        slug = self._slug(provider)
        changed = slug != self._active
        self._active = slug
        self._bucket()
        self.check_reset()
        if capacity is not None:
            self.set_capacity(capacity)
        if changed:
            self.save_state()

    def set_capacity(self, capacity: float) -> None:
        """Dynamic capacity for the active provider (its context window).

        Note the unit: capacity is expressed in the same "tokens" the cost model
        charges. At WORDS_PER_TOKEN = 10, a 2,048 capacity is ~20,480 words —
        roughly 400 ordinary messages — with compaction at ~17,000 words.
        """
        if capacity and capacity > 0:
            self._capacities[self._active] = float(capacity)

    def providers_snapshot(self) -> list[dict]:
        """Every provider's window for the cost dashboard.

        Display-only: an expired bucket reads as empty with no reset time left,
        without mutating the stored state of providers that are not active.
        """
        now = time.time()
        rows: list[dict] = []
        for slug in sorted(self._providers):
            bucket = self._providers[slug]
            last_reset = float(bucket.get("last_reset", now))
            expired = (now - last_reset) > self.RESET_WINDOW
            used = 0.0 if expired else float(bucket.get("usage", 0.0) or 0.0)
            compactions = 0 if expired else int(bucket.get("compaction_count", 0) or 0)
            capacity = float(self._capacities.get(slug, self.DEFAULT_CAPACITY))
            rows.append(
                {
                    "provider": slug,
                    "active": slug == self._active,
                    "used": round(used, 2),
                    "capacity": round(capacity, 2),
                    "percent": min(100, int((used / max(capacity, 1)) * 100)),
                    "resets_in_seconds": (
                        0 if expired else max(0, int(self.RESET_WINDOW - (now - last_reset)))
                    ),
                    "compactions": compactions,
                    "max_compactions": self.MAX_COMPACTIONS,
                }
            )
        return rows

    def load_state(self) -> None:
        try:
            path = config_manager.SETTINGS_FILE.replace("settings.json", "usage_state.json")
            if not os.path.exists(path):
                return
            with open(path, "r") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return

            providers = data.get("providers")
            if isinstance(providers, dict):
                loaded: dict[str, dict] = {}
                for slug, bucket in providers.items():
                    if not isinstance(bucket, dict):
                        continue
                    loaded[str(slug)] = {
                        "usage": float(bucket.get("usage", 0.0) or 0.0),
                        "last_reset": float(bucket.get("last_reset", time.time()) or time.time()),
                        "compaction_count": int(bucket.get("compaction_count", 0) or 0),
                    }
                if loaded:
                    self._providers = loaded
                    active = self._slug(data.get("active"))
                    self._active = active if active in loaded else "local"
                return

            # v1: one flat window, effectively the local model's usage.
            if "usage" in data or "last_reset" in data:
                self._providers = {
                    "local": {
                        "usage": float(data.get("usage", 0.0) or 0.0),
                        "last_reset": float(data.get("last_reset", time.time()) or time.time()),
                        "compaction_count": int(data.get("compaction_count", 0) or 0),
                    }
                }
                self._active = "local"
        except Exception:
            pass

    def save_state(self) -> None:
        try:
            path = config_manager.SETTINGS_FILE.replace("settings.json", "usage_state.json")
            with open(path, "w") as f:
                json.dump(
                    {
                        "version": self.STATE_VERSION,
                        "active": self._active,
                        "providers": self._providers,
                    },
                    f,
                )
        except Exception:
            pass

    def check_reset(self) -> None:
        now = time.time()
        bucket = self._bucket()
        if now - float(bucket.get("last_reset", now)) > self.RESET_WINDOW:
            bucket["usage"] = 0.0
            bucket["compaction_count"] = 0
            bucket["last_reset"] = now
            self.save_state()

    def add_usage(self, amount: float) -> None:
        self.check_reset()
        bucket = self._bucket()
        bucket["usage"] = min(float(bucket.get("usage", 0.0)) + amount, self.max_capacity)
        self.save_state()

    @property
    def usage(self) -> float:
        self.check_reset()
        return float(self._bucket().get("usage", 0.0))

    @property
    def percentage(self) -> int:
        return min(100, int((self.usage / max(self.max_capacity, 1)) * 100))

    @property
    def max_capacity(self) -> float:
        return self._capacities.get(self._active, self.DEFAULT_CAPACITY)

    @property
    def is_exhausted(self) -> bool:
        """True if current usage meets or exceeds max capacity."""
        return self.usage >= self.max_capacity

    @property
    def needs_compaction(self) -> bool:
        """True when usage has reached the 84% compaction threshold."""
        return self.percentage >= self.COMPACTION_THRESHOLD

    @property
    def compaction_count(self) -> int:
        return int(self._bucket().get("compaction_count", 0))

    @property
    def compactions_remaining(self) -> int:
        return max(0, self.MAX_COMPACTIONS - self.compaction_count)

    def can_compact(self) -> bool:
        """Compaction allowed until 5 have been used in this window."""
        return self.compaction_count < self.MAX_COMPACTIONS

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
    # bucket internals directly. Each one persists, exactly like real usage.
    def force_usage(self, amount: float) -> None:
        self._bucket()["usage"] = max(0.0, min(float(amount), self.max_capacity))
        self.save_state()

    def force_percent(self, percent: float) -> None:
        self.force_usage(self.max_capacity * max(0.0, min(100.0, percent)) / 100.0)

    def force_reset(self) -> None:
        bucket = self._bucket()
        bucket["usage"] = 0.0
        bucket["compaction_count"] = 0
        bucket["last_reset"] = time.time()
        self.save_state()

    def expire_window(self) -> None:
        """Pretend the 12-hour window has elapsed, so reset logic can be seen."""
        self._bucket()["last_reset"] = time.time() - self.RESET_WINDOW - 1
        self.save_state()

    def run_compaction(self) -> None:
        """Count the compaction and reset usage back to 0%."""
        bucket = self._bucket()
        bucket["compaction_count"] = int(bucket.get("compaction_count", 0)) + 1
        bucket["usage"] = 0.0
        self.check_reset()
        self.save_state()

    def time_until_reset(self) -> str:
        """Return human-readable time until the 12-hour usage window resets."""
        remaining = self.RESET_WINDOW - (time.time() - float(self._bucket().get("last_reset", time.time())))
        if remaining <= 0:
            return "any moment"
        hours = int(remaining // 3600)
        minutes = int((remaining % 3600) // 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"
