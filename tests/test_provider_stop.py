"""Stopping an API-provider reply must release the generation slot.

The bug: the worker only noticed a stop when the provider sent another chunk.
A quiet stream kept `is_generating` true, so the next prompt or new chat was
refused with "A response is already being generated." Local models were fine
because their backend stop is cooperative.
"""

from __future__ import annotations

import copy
import threading
import time
import unittest
from unittest import mock

from dizel_core.chat_manager import ChatManager
from dizel_core.config_manager import ConfigManager
from dizel_core.providers import BaseProvider, ProviderError


class BlockingProvider(BaseProvider):
    """Streams one token, then holds the socket until cancel() is called."""

    label = "Blocking"

    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.cancelled = threading.Event()

    def stream_events(self, messages, model, key="", temperature=0.0, max_tokens=8, **_):
        self.started.set()
        try:
            yield ("content", "partial ")
            self.cancelled.wait(5)
            # requests raises when a response is closed under iter_lines();
            # Providers convert that to ProviderError. Mirror it here.
            raise ProviderError("stream closed")
        finally:
            self.cancelled.set()

    def cancel(self) -> None:
        super().cancel()
        self.cancelled.set()


class QuickProvider(BaseProvider):
    label = "Quick"

    def stream_events(self, messages, model, key="", temperature=0.0, max_tokens=8, **_):
        yield ("content", "second reply")


class ProviderStopTests(unittest.TestCase):
    def _manager(self) -> ChatManager:
        defaults = copy.deepcopy(ConfigManager._DEFAULTS)
        with mock.patch.object(
            ConfigManager, "load", classmethod(lambda cls: copy.deepcopy(defaults))
        ):
            return ChatManager()

    def _run_provider(self, manager: ChatManager, prompt: str):
        tokens: list[str] = []
        errors: list[str] = []
        done = threading.Event()
        result: list[str] = []

        manager.send_message(
            prompt,
            [],
            lambda piece: tokens.append(piece),
            lambda text: (result.append(text), done.set()),
            lambda message: (errors.append(message), done.set()),
        )
        return tokens, errors, done, result

    def test_cancel_closes_the_tracked_response(self):
        class Response:
            closed = False

            def close(self) -> None:
                self.closed = True

        provider = BaseProvider()
        response = Response()
        provider._track_response(response)
        provider.cancel()
        self.assertTrue(response.closed)

    def test_stop_right_after_send_is_not_erased_by_the_worker(self):
        """The worker used to reset the stop flag when its thread began."""
        manager = self._manager()
        provider = BlockingProvider()
        manager._provider_slug = "blocking"
        manager._api_provider = provider
        manager._api_model = "test-model"
        manager._api_key = "test-key"

        _tokens, errors, done, _result = self._run_provider(manager, "race")
        manager.stop_generation()

        self.assertTrue(done.wait(2.0), "worker never observed the early stop")
        self.assertEqual(errors, [])
        self.assertFalse(manager.is_generating)

    def test_stop_frees_slot_and_next_prompt_is_accepted(self):
        manager = self._manager()
        first = BlockingProvider()
        manager._provider_slug = "blocking"
        manager._api_provider = first
        manager._api_model = "test-model"
        manager._api_key = "test-key"

        tokens, errors, done, result = self._run_provider(manager, "first")
        self.assertTrue(first.started.wait(1.0), "provider never started")
        self.assertTrue(manager.is_generating)

        started = time.monotonic()
        manager.stop_generation()
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertTrue(done.wait(1.0), "worker did not finish after stop")
        self.assertFalse(manager.is_generating)
        self.assertEqual(errors, [])
        self.assertEqual(result, ["partial "])
        self.assertEqual(tokens, ["partial "])

        # The exact user story: send again immediately, and start a new chat.
        second = QuickProvider()
        manager._api_provider = second
        tokens2, errors2, done2, result2 = self._run_provider(manager, "second")
        self.assertTrue(done2.wait(2.0), "second prompt was never accepted")
        self.assertEqual(errors2, [])
        self.assertEqual("".join(tokens2), "second reply")
        self.assertEqual(result2, ["second reply"])
        self.assertFalse(manager.is_generating)

        # A new chat clears history; generation must not be claimed.
        manager.new_session()
        self.assertFalse(manager.is_generating)


if __name__ == "__main__":
    unittest.main()
