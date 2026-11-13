"""A continuation is not a new turn.

The bug this guards: "Continue" sent a synthetic prompt — *"Continue exactly where
you stopped…"* — as an ordinary message. The client joined the two halves into one
answer, but the server persisted the synthetic turn, so reloading a session showed
an answer plus a stray instruction. The fix threads a `continuation` flag from the
route down to `_commit_reply`, which appends to the last assistant message instead
of pushing a new one.

These tests cover the decision itself, which is where the bug lived. They need no
model, no sidecar and no network.
"""

from __future__ import annotations

import inspect
import unittest

from dizel_core.chat_manager import ChatManager


def manager_with(history):
    """A ChatManager holding `history`, with no backend loaded."""
    manager = ChatManager()
    manager.history = history
    return manager


class CommitReplyTests(unittest.TestCase):
    def test_continuation_appends_to_the_last_assistant_message(self):
        manager = manager_with(
            [
                {"role": "user", "content": "Tell me about the Seine."},
                {"role": "assistant", "content": "The Seine rises in Burgundy and flows"},
            ]
        )

        manager._commit_reply("northwest through Paris.", True)

        self.assertEqual(len(manager.history), 2, "a continuation must not add a turn")
        self.assertEqual(
            manager.history[-1]["content"],
            "The Seine rises in Burgundy and flows\n\nnorthwest through Paris.",
        )

    def test_a_normal_reply_adds_a_turn(self):
        manager = manager_with(
            [
                {"role": "user", "content": "Tell me about the Seine."},
                {"role": "assistant", "content": "It flows northwest."},
                {"role": "user", "content": "And the Loire?"},
            ]
        )

        manager._commit_reply("Longer, and wilder.", False)

        self.assertEqual(len(manager.history), 4)
        self.assertEqual(manager.history[-1]["content"], "Longer, and wilder.")
        self.assertEqual(manager.history[-1]["role"], "assistant")

    def test_continuation_without_a_prior_reply_starts_one(self):
        """The guard matters: appending into a *user* turn would corrupt it."""
        manager = manager_with([{"role": "user", "content": "Say something."}])

        manager._commit_reply("Something.", True)

        self.assertEqual(len(manager.history), 2)
        self.assertEqual(manager.history[0]["content"], "Say something.")
        self.assertEqual(manager.history[1]["content"], "Something.")

    def test_continuation_on_empty_history_starts_one(self):
        manager = manager_with([])

        manager._commit_reply("First words.", True)

        self.assertEqual(manager.history, [{"role": "assistant", "content": "First words."}])

    def test_repeated_continuations_keep_joining(self):
        manager = manager_with(
            [
                {"role": "user", "content": "Count."},
                {"role": "assistant", "content": "One"},
            ]
        )

        manager._commit_reply("two", True)
        manager._commit_reply("three", True)

        self.assertEqual(len(manager.history), 2)
        self.assertEqual(manager.history[-1]["content"], "One\n\ntwo\n\nthree")

    def test_continuation_strips_trailing_whitespace_at_the_seam(self):
        manager = manager_with(
            [
                {"role": "user", "content": "Hi."},
                {"role": "assistant", "content": "Hello   \n"},
            ]
        )

        manager._commit_reply("  again.  ", True)

        self.assertEqual(manager.history[-1]["content"], "Hello\n\nagain.")


class PlumbingTests(unittest.TestCase):
    """The flag had four hops to travel. A missed one would not raise — it would
    silently behave like an ordinary send, so each hop is asserted by signature."""

    def test_commit_reply_takes_the_flag(self):
        params = inspect.signature(ChatManager._commit_reply).parameters
        self.assertIn("continuation", params)

    def test_send_message_accepts_the_flag(self):
        params = inspect.signature(ChatManager.send_message).parameters
        self.assertIn("continuation", params)
        self.assertFalse(params["continuation"].default)

    def test_send_with_tools_accepts_the_flag(self):
        params = inspect.signature(ChatManager.send_with_tools).parameters
        self.assertIn("continuation", params)
        self.assertFalse(params["continuation"].default)

    def test_generate_workers_accept_the_flag(self):
        for name in ("_generate_via_api", "_generate_worker"):
            with self.subTest(worker=name):
                params = inspect.signature(getattr(ChatManager, name)).parameters
                self.assertIn("continuation", params)

    def test_request_model_defaults_to_an_ordinary_send(self):
        from api.routes.chat import ChatRequest

        self.assertFalse(ChatRequest(user_text="hello").continuation)
        self.assertTrue(ChatRequest(user_text="hello", continuation=True).continuation)

    def test_service_signature_accepts_the_flag(self):
        from api.services.chat_service import ChatService

        params = inspect.signature(ChatService.stream_chat).parameters
        self.assertIn("continuation", params)
        self.assertFalse(params["continuation"].default)


if __name__ == "__main__":
    unittest.main()
