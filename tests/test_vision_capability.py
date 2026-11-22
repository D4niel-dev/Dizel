"""Vision capability comes from the provider when it can answer.

The paperclip is gated on `model_info.vision`. Before this, that flag was a regex
over the model name, so a vision model with an unfamiliar name was silently
blocked and a text model with a promising name was let through.
"""

from __future__ import annotations

import unittest

from dizel_core.chat_manager import ChatManager
from dizel_core.providers import (
    OllamaProvider,
    probe_vision,
    resolve_vision,
)


class _Response:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


def _ollama(payload=None, status=200, error=None):
    """An OllamaProvider whose /api/show is stubbed."""
    provider = OllamaProvider("http://127.0.0.1:11434")

    def post(*_args, **_kwargs):
        if error is not None:
            raise error
        return _Response(payload, status)

    provider._session.post = post  # type: ignore[assignment]
    return provider


class ShowModelTests(unittest.TestCase):
    def test_reports_the_daemons_capabilities(self):
        provider = _ollama({"capabilities": ["completion", "vision"]})
        self.assertEqual(provider.show_model("llava"), ["completion", "vision"])

    def test_missing_capabilities_key_is_none(self):
        provider = _ollama({"model_info": {}})
        self.assertIsNone(provider.show_model("mystery"))

    def test_non_200_is_none(self):
        provider = _ollama({"capabilities": ["vision"]}, status=500)
        self.assertIsNone(provider.show_model("llava"))

    def test_unparseable_body_is_none(self):
        provider = OllamaProvider("http://127.0.0.1:11434")
        provider._session.post = lambda *a, **k: _Response(None)  # type: ignore[assignment]
        self.assertIsNone(provider.show_model("llava"))


class ProbeVisionTests(unittest.TestCase):
    def test_vision_wins_over_an_unfamiliar_name(self):
        # The regex has never heard of this model; the daemon has.
        provider = _ollama({"capabilities": ["completion", "vision"]})
        self.assertIs(probe_vision(provider, "some-custom-vlm"), True)

    def test_no_vision_wins_over_a_promising_name(self):
        # `llava` matches the regex, but this build cannot see.
        provider = _ollama({"capabilities": ["completion"]})
        self.assertIs(probe_vision(provider, "llava"), False)

    def test_connection_error_is_unknown_not_false(self):
        provider = _ollama(error=ConnectionError("daemon down"))
        self.assertIsNone(probe_vision(provider, "llava"))

    def test_provider_without_the_endpoint_is_unknown(self):
        self.assertIsNone(probe_vision(object(), "gpt-4o"))
        self.assertIsNone(probe_vision(None, "gpt-4o"))


class ResolveVisionTests(unittest.TestCase):
    def test_falls_back_to_the_name_heuristic(self):
        self.assertIs(resolve_vision("ollama", "llava", None), True)
        self.assertIs(resolve_vision("ollama", "qwen2.5:7b", None), False)

    def test_a_cached_answer_beats_the_heuristic(self):
        self.assertIs(resolve_vision("ollama", "llava", False), False)
        self.assertIs(resolve_vision("ollama", "qwen2.5:7b", True), True)


class _CountingProvider:
    """Stands in for OllamaProvider; counts how often it is asked."""

    def __init__(self, capabilities):
        self.capabilities = capabilities
        self.calls = 0

    def show_model(self, _model):
        self.calls += 1
        return self.capabilities


class ManagerCacheTests(unittest.TestCase):
    def _manager(self, provider, model="llava"):
        manager = ChatManager.__new__(ChatManager)
        manager._provider_slug = "ollama"
        manager._api_model = model
        manager._api_provider = provider
        manager._vision_flags = {}
        return manager

    def test_model_info_reads_the_cached_answer(self):
        provider = _CountingProvider(["completion", "vision"])
        manager = self._manager(provider)

        manager._probe_vision()
        self.assertEqual(provider.calls, 1)

        # Status polls hit model_info repeatedly; none of them may re-probe.
        self.assertIs(manager.model_info["vision"], True)
        self.assertIs(manager.model_info["vision"], True)
        self.assertEqual(provider.calls, 1)

    def test_an_unknown_answer_leaves_the_cache_empty(self):
        provider = _CountingProvider(None)
        manager = self._manager(provider)

        manager._probe_vision()
        self.assertEqual(manager._vision_flags, {})
        # Falls back to the regex, which knows `llava`.
        self.assertIs(manager.model_info["vision"], True)

    def test_local_provider_never_probes(self):
        provider = _CountingProvider(["vision"])
        manager = self._manager(provider)
        manager._provider_slug = "local"

        manager._probe_vision()
        self.assertEqual(provider.calls, 0)
        self.assertEqual(manager._vision_flags, {})


if __name__ == "__main__":
    unittest.main()
