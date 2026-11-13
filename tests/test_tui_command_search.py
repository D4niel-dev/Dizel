"""Fuzzy command search and slash-drawer paging stay usable with 60+ skills."""

from __future__ import annotations

import unittest

from inference.tui.commands.matching import match_score
from inference.tui.commands.registry import Command, CommandRegistry
from inference.tui.panels.slash_drawer import MAX_ROWS, SlashDrawer


class Dummy(Command):
    def __init__(self, name, aliases=(), category="General", help_text="", source=None):
        self.name = name
        self.aliases = list(aliases)
        self.category = category
        self.help_text = help_text
        if source:
            self.skill = type("SkillRef", (), {"source": source})()

    async def execute(self, app, invocation):  # pragma: no cover - search only
        return ""


class CommandSearchTests(unittest.TestCase):
    def setUp(self):
        self.registry = CommandRegistry()
        self.registry.register(Dummy("summarize", ["tldr"], "Writing"))
        self.registry.register(Dummy("systematic-debugging", [], "Coding"))
        self.registry.register(Dummy("sql", ["query"], "Coding"))
        self.registry.register(
            Dummy("frontend-design", [], "Local", "Design system guidance.", "project")
        )

    def test_prefix_beats_fuzzy_and_alias_matches(self):
        names = [command.name for command in self.registry.search("sum")]
        self.assertEqual(names[0], "summarize")
        self.assertEqual(self.registry.search("tldr")[0].name, "summarize")
        self.assertEqual(self.registry.search("query")[0].name, "sql")

    def test_gapped_subsequence_matches(self):
        self.assertEqual(self.registry.search("smz")[0].name, "summarize")
        self.assertEqual(self.registry.search("sysdbg")[0].name, "systematic-debugging")

    def test_no_match_and_empty_query(self):
        self.assertEqual(self.registry.search("zzzz"), [])
        self.assertEqual(len(self.registry.search("")), 4)

    def test_scores_rank_exact_before_gapped(self):
        self.assertLess(match_score("summarize", "summarize"), match_score("summarize", "smz"))


class DrawerTests(unittest.TestCase):
    def _drawer(self, count=20):
        drawer = SlashDrawer()
        matches = [Dummy(f"cmd{index}") for index in range(count)]
        drawer.show(matches, "")
        return drawer

    def test_header_reports_total_matches(self):
        drawer = self._drawer(20)
        self.assertIn("20 matches", drawer._header(96))
        self.assertEqual(len(drawer._matches), 20)

    def test_page_moves_selection_and_window(self):
        drawer = self._drawer(20)
        drawer.page(1)
        self.assertEqual(drawer._index, MAX_ROWS)
        self.assertGreater(drawer._window_start, 0)
        drawer.move(-1)
        self.assertEqual(drawer._index, MAX_ROWS - 1)

    def test_source_is_shown_before_description(self):
        drawer = SlashDrawer()
        command = Dummy("frontend-design", category="Local", help_text="Design.", source="project")
        row = drawer._row(command, False, 96)
        self.assertIn("local · Design.", row)


if __name__ == "__main__":
    unittest.main()
