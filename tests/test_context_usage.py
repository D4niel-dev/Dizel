"""The context meter must count the whole composed system prompt.

`system_content()` joins the user's base prompt, the active mode clause and the
project rules. Counting only `system_prompt` silently dropped the last two, so the
meter under-reported exactly when a mode clause was long or a project file existed.
"""

from __future__ import annotations

import unittest
from unittest import mock

from api.services.chat_service import ChatService


class _Manager:
    """Stands in for ChatManager; counts whitespace-separated words."""

    def __init__(self, base="", mode="", project=""):
        self.system_prompt = base
        self.mode_prompt = mode
        self.project_context = project
        self.effective_ctx_len = 2048
        self.last_budget = 0
        self.last_trimmed = 0
        self.exact_counts = True
        self.last_task = ""

    def count_tokens(self, text):
        return len(str(text).split())


class SystemBlockTests(unittest.TestCase):
    def test_splits_the_three_parts(self):
        manager = _Manager(base="a b c", mode="d e", project="f")
        blocks = ChatService._system_blocks(manager)
        self.assertEqual(blocks, {"base": 3, "mode": 2, "verbosity": 0, "project": 1})

    def test_missing_parts_are_zero_not_absent(self):
        manager = _Manager(base="a b")
        blocks = ChatService._system_blocks(manager)
        self.assertEqual(blocks, {"base": 2, "mode": 0, "verbosity": 0, "project": 0})

    def test_a_manager_without_the_new_attributes_still_works(self):
        class Bare:
            system_prompt = "one two three"

            def count_tokens(self, text):
                return len(str(text).split())

        blocks = ChatService._system_blocks(Bare())
        self.assertEqual(blocks["base"], 3)
        self.assertEqual(blocks["verbosity"], 0)
        self.assertEqual(blocks["mode"], 0)
        self.assertEqual(blocks["project"], 0)


class ContextUsageTests(unittest.TestCase):
    def _usage(self, manager, session_id=None):
        with mock.patch.object(ChatService, "get_manager", return_value=manager):
            return ChatService.context_usage(session_id)

    def test_system_is_the_sum_of_its_blocks(self):
        manager = _Manager(base="a b c", mode="d e", project="f g h i")
        usage = self._usage(manager)
        blocks = usage["system_blocks"]
        self.assertEqual(usage["system"], sum(blocks.values()))
        self.assertEqual(usage["system"], 9)

    def test_used_counts_the_mode_clause_and_project(self):
        manager = _Manager(base="a", mode="b c", project="d e f")
        usage = self._usage(manager)
        # Before the split this reported 1: the mode clause and project vanished.
        self.assertEqual(usage["used"], 6)
        self.assertEqual(usage["conversation"], 0)

    def test_percent_stays_within_the_window(self):
        manager = _Manager(base=" ".join(["w"] * 1024))
        usage = self._usage(manager)
        self.assertEqual(usage["total"], 2048)
        self.assertEqual(usage["percent"], 50)


if __name__ == "__main__":
    unittest.main()
