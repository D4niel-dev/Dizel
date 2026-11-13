import dataclasses
from typing import Any, List

from inference.tui.commands.parser import CommandInvocation
from inference.tui.commands.registry import Command, registry

from dizel_core import skills as skills_core

# Matches the GUI threshold: flag prompts large enough to pressure a small
# local context window.
LARGE_SKILL_CHARS = 4000


class SkillCommand(Command):
    """One registered prompt template, invoked as /name."""

    category = "Skills"

    def __init__(self, skill: skills_core.Skill) -> None:
        self.skill = skill
        self.name = skill.name
        self.help_text = skill.description
        self.usage = f"/{skill.name} [input]"
        self.palette_hint = f"/{skill.name} "
        self.aliases = list(skill.aliases)
        self.examples = [f"/{skill.name} <text to work on>"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        prompt = skills_core.expand_skill(self.skill, " ".join(invocation.args))
        app.insert_prompt(prompt)
        if len(prompt) > LARGE_SKILL_CHARS:
            return (
                f"Inserted /{self.name} into the composer — large "
                f"(~{len(prompt) // 4} tokens); trim it before sending."
            )
        return (
            f"Inserted /{self.name} into the composer — "
            "edit it if needed, then press enter to send."
        )


class SkillsCommand(Command):
    name = "skills"
    help_text = "Browse prompt templates by category, or reload them from disk."
    category = "Core"
    usage = "/skills [reload|category]"
    palette_hint = "/skills "
    examples = ["/skills", "/skills writing", "/skills reload"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        action = invocation.args[0].lower() if invocation.args else ""
        if action == "reload":
            count = register_skills()
            return f"Reloaded {count} skill command(s) from disk."

        skills = skills_core.list_skills()
        if not skills:
            return (
                "No skills found. Add Markdown files to "
                f"{skills_core.USER_DIR}, then run /skills reload."
            )

        groups = skills_core.group_skills(skills)

        if action:
            needle = action.replace("-", " ").replace("_", " ").strip()
            category = next((name for name, _ in groups if name.lower() == needle), None)
            if category is None:
                names = ", ".join(name for name, _ in groups)
                return f"Unknown skill category: {action}\nCategories: {names}"
            entries = next(entries for name, entries in groups if name == category)
            lines = [f"{category.upper()}  ·  {len(entries)}", ""]
            for skill in entries:
                aliases = ", ".join(f"/{alias}" for alias in skill.aliases)
                lines.append(f"/{skill.name:<14}{skill.description}")
                detail = {"builtin": "built-in", "user": "user", "project": "project"}.get(
                    skill.source, "user"
                )
                if aliases:
                    detail = f"{detail}  ·  aliases: {aliases}"
                lines.append(f"{'':<15}{detail}")
            return "\n".join(lines)

        lines = [f"Skills · {len(skills)}", ""]
        for category, entries in groups:
            if len(entries) > 10:
                # Large local packs would bury the summary; the category view
                # lists them on demand.
                slug = category.lower().replace(" ", "-")
                lines.append(f"{category.upper()} {len(entries)} · /skills {slug}")
                continue
            names = ", ".join(f"/{skill.name}" for skill in entries)
            lines.append(f"{category.upper()} {len(entries)} · {names}")
        lines.extend(
            [
                "",
                "Type /skills <category> for descriptions and aliases.",
                "Type / to browse and filter every command.",
                f"User skills: {skills_core.USER_DIR}",
                f"Project skills: {skills_core.project_skills_dir()} "
                "(folders with a SKILL.md)",
                "Run /skills reload after adding a skill.",
            ]
        )
        return "\n".join(lines)


def register_skills() -> int:
    """(Re)register one command per skill, without shadowing built-ins."""
    for command in [entry for entry in registry.list_all() if isinstance(entry, SkillCommand)]:
        registry.unregister(command)

    count = 0
    for skill in skills_core.list_skills():
        if registry.lookup(skill.name):
            continue
        allowed: List[str] = []
        for alias in skill.aliases:
            if alias == skill.name or alias in allowed:
                continue
            if registry.lookup(alias):
                continue
            allowed.append(alias)
        command = SkillCommand(dataclasses.replace(skill, aliases=tuple(allowed)))
        registry.register(command)
        count += 1
    return count
