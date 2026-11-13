"""Skills are shared prompt templates; both surfaces must see the same catalog."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dizel_core import skills as skills_core

BUILTIN_NAMES = {"brainstorm", "explain", "review", "rewrite", "summarize"}


class SkillCatalogTests(unittest.TestCase):
    def test_builtins_ship_with_the_app(self):
        catalog = {
            skill.name: skill
            for skill in skills_core.list_skills(user_dir="", project_dir="")
        }
        self.assertTrue(BUILTIN_NAMES <= set(catalog))
        self.assertGreaterEqual(len(catalog), 24)
        self.assertTrue(all(skill.builtin for skill in catalog.values()))
        self.assertTrue(all(skill.description for skill in catalog.values()))
        self.assertIn("{{input}}", catalog["summarize"].body)
        self.assertTrue(
            {"Writing", "Thinking", "Coding", "Learning", "Work", "Data"}
            <= {skill.category for skill in catalog.values()}
        )
        self.assertEqual(
            skills_core.get_skill("grammar", user_dir="", project_dir="").name, "proofread"
        )

    def test_project_folder_skills_are_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp, ".agent", "skills")
            api = root / "api-patterns"
            api.mkdir(parents=True)
            (api / "SKILL.md").write_text(
                "---\nname: api-patterns\ndescription: API design principles.\n---\n"
                "Use REST for public APIs.\n",
                encoding="utf-8",
            )
            (api / "reference.md").write_text("supplementary notes", encoding="utf-8")
            notes = root / "notes"
            notes.mkdir()
            (notes / "SKILL.md").write_text(
                "---\ndescription: Note-taking workflow.\n---\nWrite the notes.\n",
                encoding="utf-8",
            )
            (root / "doc.md").write_text("guide, not a skill", encoding="utf-8")

            catalog = {
                skill.name: skill
                for skill in skills_core.list_skills(user_dir="", project_dir=str(root))
            }
            self.assertIn("api-patterns", catalog)
            self.assertEqual(catalog["api-patterns"].source, "project")
            self.assertEqual(catalog["api-patterns"].category, "Local")
            self.assertFalse(catalog["api-patterns"].builtin)
            self.assertIn("notes", catalog)
            self.assertEqual(catalog["notes"].title, "Notes")
            self.assertNotIn("reference", catalog)
            self.assertNotIn("doc", catalog)

    def test_project_skill_overrides_builtin_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp, "summarize")
            summary.mkdir(parents=True)
            (summary / "SKILL.md").write_text(
                "---\nname: summarize\ndescription: Local summary.\n---\nLocal body.\n",
                encoding="utf-8",
            )
            catalog = {
                skill.name: skill
                for skill in skills_core.list_skills(user_dir="", project_dir=tmp)
            }
            self.assertEqual(catalog["summarize"].source, "project")
            self.assertEqual(catalog["summarize"].body, "Local body.")

    def test_categories_follow_the_shared_reading_order(self):
        catalog = skills_core.list_skills(user_dir="", project_dir="")
        groups = skills_core.group_skills(catalog)
        self.assertEqual(
            [label for label, _ in groups][:6],
            ["Writing", "Thinking", "Coding", "Learning", "Work", "Data"],
        )
        self.assertEqual(sum(len(entries) for _, entries in groups), len(catalog))

    def test_user_skill_overrides_builtin(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "summarize.md").write_text(
                "---\n"
                "name: summarize\n"
                "title: My summary\n"
                "description: Mine, not the built-in.\n"
                "aliases: [brief]\n"
                "category: Coding\n"
                "---\n"
                "Summarize my way: {{input}}\n",
                encoding="utf-8",
            )
            catalog = {
                skill.name: skill
                for skill in skills_core.list_skills(user_dir=tmp, project_dir="")
            }
            override = catalog["summarize"]
            self.assertFalse(override.builtin)
            self.assertEqual(override.title, "My summary")
            self.assertEqual(override.aliases, ("brief",))
            self.assertEqual(override.category, "Coding")
            self.assertEqual(
                skills_core.get_skill("brief", user_dir=tmp, project_dir="").name,
                "summarize",
            )

    def test_filename_becomes_name_and_first_line_becomes_description(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "Quick Fix.md").write_text("Fix the thing.\n", encoding="utf-8")
            skill = skills_core.get_skill("quick-fix", user_dir=tmp, project_dir="")
            self.assertIsNotNone(skill)
            self.assertEqual(skill.name, "quick-fix")
            self.assertEqual(skill.description, "Fix the thing.")
            self.assertEqual(skill.aliases, ())

    def test_unknown_skill_is_none(self):
        self.assertIsNone(
            skills_core.get_skill("does-not-exist", user_dir="", project_dir="")
        )

    def test_expand_replaces_input_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "echo.md").write_text(
                "---\nname: echo\ndescription: Echo.\n---\nWork on: {{input}}\n",
                encoding="utf-8",
            )
            skill = skills_core.get_skill("echo", user_dir=tmp, project_dir="")
            self.assertIn("hello", skills_core.expand_skill(skill, " hello "))
            self.assertIn(skills_core.EMPTY_INPUT, skills_core.expand_skill(skill))

    def test_expand_appends_input_without_token(self):
        skill = skills_core.Skill(name="x", title="X", description="", body="Do the thing.")
        self.assertEqual(skills_core.expand_skill(skill), "Do the thing.")
        self.assertEqual(
            skills_core.expand_skill(skill, "abc"), "Do the thing.\n\nInput:\nabc"
        )


class TuiSkillCommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from inference.tui.commands.builtins import register_builtins
        from inference.tui.commands.builtins.skills import register_skills

        register_builtins()
        self.register_skills = register_skills
        self.fake = skills_core.Skill(
            name="unit-skill",
            title="Unit skill",
            description="Used by the test suite.",
            body="Apply this to {{input}}",
            aliases=("us", "help"),
        )

    def tearDown(self):
        # Put the real catalog back for any later test in this process.
        self.register_skills()

    def test_registration_skips_shadowed_names_and_aliases(self):
        from inference.tui.commands.registry import registry

        with mock.patch.object(skills_core, "list_skills", return_value=[self.fake]):
            self.assertEqual(self.register_skills(), 1)

        command = registry.lookup("unit-skill")
        self.assertIsNotNone(command)
        self.assertEqual(command.aliases, ["us"])
        self.assertEqual(registry.lookup("us").name, "unit-skill")
        self.assertEqual(registry.lookup("help").name, "help")

    async def test_execute_inserts_expanded_prompt(self):
        from inference.tui.commands.parser import parse_command
        from inference.tui.commands.registry import registry

        class App:
            inserted = ""

            def insert_prompt(self, text: str) -> None:
                self.inserted = text

        with mock.patch.object(skills_core, "list_skills", return_value=[self.fake]):
            self.register_skills()

        app = App()
        result = await registry.lookup("unit-skill").execute(
            app, parse_command("/unit-skill the payload")
        )
        self.assertIn("Apply this to the payload", app.inserted)
        self.assertIn("/unit-skill", result)

    async def test_large_skill_warns_before_sending(self):
        from inference.tui.commands.builtins.skills import SkillCommand
        from inference.tui.commands.parser import parse_command

        class App:
            inserted = ""

            def insert_prompt(self, text: str) -> None:
                self.inserted = text

        skill = skills_core.Skill(
            name="big-skill",
            title="Big skill",
            description="Large.",
            body="x" * 5000,
        )
        app = App()
        result = await SkillCommand(skill).execute(app, parse_command("/big-skill"))
        self.assertIn("large", result)
        self.assertEqual(len(app.inserted), 5000)

    async def test_skills_command_lists_and_reloads(self):
        from inference.tui.commands.parser import parse_command
        from inference.tui.commands.registry import registry

        command = registry.lookup("skills")
        listing = await command.execute(None, parse_command("/skills"))
        self.assertIn("Skills", listing)
        self.assertIn("WRITING", listing)
        self.assertIn("CODING", listing)
        self.assertIn("brainstorm", listing)
        self.assertIn("User skills", listing)
        self.assertIn("Project skills", listing)
        self.assertNotIn("**", listing)
        # A command result must stay a summary, not a 150-line transcript.
        self.assertLess(len(listing.splitlines()), 30)

        writing = await command.execute(None, parse_command("/skills writing"))
        self.assertIn("WRITING", writing)
        self.assertIn("/summarize", writing)

        unknown = await command.execute(None, parse_command("/skills nope"))
        self.assertIn("Unknown skill category", unknown)

        reloaded = await command.execute(None, parse_command("/skills reload"))
        self.assertIn("skill command", reloaded)


class SkillRouteTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        from api.main import app

        self.tmp = tempfile.TemporaryDirectory()
        self.patch = mock.patch.object(skills_core, "USER_DIR", self.tmp.name)
        self.patch.start()
        self.project_patch = mock.patch.object(
            skills_core, "project_skills_dir", return_value=""
        )
        self.project_patch.start()
        # No context manager: avoids the API lifespan, which would connect MCP
        # servers from the machine's real settings.
        self.client = TestClient(app)

    def tearDown(self):
        self.project_patch.stop()
        self.patch.stop()
        self.tmp.cleanup()

    def test_list_returns_builtins(self):
        response = self.client.get("/api/skills/")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        names = {skill["name"] for skill in payload["skills"]}
        self.assertTrue(BUILTIN_NAMES <= names)
        self.assertEqual(payload["user_dir"], self.tmp.name)
        self.assertTrue(all("aliases" in skill for skill in payload["skills"]))
        categories = {skill["category"] for skill in payload["skills"]}
        self.assertTrue({"Writing", "Thinking", "Coding"} <= categories)
        self.assertTrue(all("source" in skill for skill in payload["skills"]))
        self.assertEqual(payload["project_dir"], "")

    def test_expand_by_alias(self):
        response = self.client.post(
            "/api/skills/expand", json={"name": "tldr", "args": "The fox jumped."}
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["name"], "summarize")
        self.assertIn("The fox jumped.", payload["prompt"])

    def test_unknown_skill_is_404(self):
        response = self.client.post(
            "/api/skills/expand", json={"name": "nope", "args": ""}
        )
        self.assertEqual(response.status_code, 404)
