"""Per-provider context/usage limits and their settings helpers."""

from dizel_core import provider_limits


def test_known_providers_get_their_own_defaults():
    assert provider_limits.resolve("openai") == 131072
    assert provider_limits.resolve("anthropic") == 200000
    assert provider_limits.resolve("google") == 1048576
    assert provider_limits.resolve("ai21") == 262144
    assert provider_limits.resolve("ollama") == 32768
    assert provider_limits.resolve("custom") == 32768


def test_unknown_provider_uses_the_fallback():
    assert provider_limits.resolve("mystery") == provider_limits.FALLBACK


def test_override_wins_over_the_default():
    assert provider_limits.resolve("openai", overrides={"openai": 4096}) == 4096


def test_local_uses_the_checkpoint_context():
    assert provider_limits.resolve("local", model_ctx_len=2048) == 2048
    assert provider_limits.resolve("local") == 2048


def test_invalid_overrides_fall_back():
    assert provider_limits.resolve("openai", overrides={"openai": 128}) == provider_limits.MIN_LIMIT
    assert provider_limits.resolve("openai", overrides={"openai": "nope"}) == 131072
    assert provider_limits.resolve("openai", overrides={"openai": 0}) == 131072
    assert provider_limits.resolve("openai", overrides="junk") == 131072


def test_set_override_writes_a_clean_map():
    api = {}
    provider_limits.set_override(api, "openai", 64000)
    assert api["provider_limits"] == {"openai": 64000}

    provider_limits.set_override(api, "openai", 0)
    assert api["provider_limits"] == {}


def test_get_override_reads_only_ints():
    api = {"provider_limits": {"openai": 5000, "groq": "x"}}
    assert provider_limits.get_override(api, "openai") == 5000
    assert provider_limits.get_override(api, "groq") is None
    assert provider_limits.get_override({}, "openai") is None


def test_effective_ctx_len_follows_the_active_provider():
    import threading

    from dizel_core.chat_manager import ChatManager

    class FakeBackend:
        is_ready = True
        model_info = {"ctx_len": 2048}

    manager = ChatManager.__new__(ChatManager)
    manager._backend_lock = threading.RLock()
    manager._provider_slug = "openai"
    manager._api_provider = None
    manager._api_model = ""
    manager._backend = None
    manager._cfg = {"api_router": {}}
    assert manager.effective_ctx_len == 131072

    manager._cfg = {"api_router": {"provider_limits": {"openai": 4096}}}
    assert manager.effective_ctx_len == 4096

    manager._provider_slug = "local"
    manager._backend = FakeBackend()
    assert manager.effective_ctx_len == 2048

    manager._backend = None
    assert manager.effective_ctx_len == 2048
