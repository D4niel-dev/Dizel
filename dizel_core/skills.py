"""Prompt skills — reusable instruction templates invoked as slash commands.

A skill is a Markdown file with a small frontmatter block:

    ---
    name: summarize
    title: Summarize
    description: Condense a passage into key points.
    aliases: summary, tldr
    ---
    Instructions, with {{input}} where the user's text goes.

Built-ins ship in ``dizel_core/skills_builtin``. User files live in
``<config dir>/skills``. Project skills live in ``<project>/.agent/skills`` as
folders with a ``SKILL.md`` (the Antigravity layout); later sources replace
earlier ones with the same name.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from dizel_core.config_manager import SETTINGS_FILE

_HERE = os.path.dirname(os.path.abspath(__file__))
BUILTIN_DIR = os.path.join(_HERE, "skills_builtin")
USER_DIR = os.path.join(os.path.dirname(SETTINGS_FILE), "skills")

INPUT_TOKEN = "{{input}}"
EMPTY_INPUT = "Paste the text here, then send."

CATEGORY_ORDER = (
    "Writing",
    "Thinking",
    "Coding",
    "Learning",
    "Work",
    "Data",
    "Local",
    "General",
)

_FRONTMATTER = re.compile(r"^\ufeff?---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|$)", re.S)
_SLUG = re.compile(r"[^a-z0-9-]+")


@dataclass(frozen=True)
class Skill:
    name: str
    title: str
    description: str
    body: str
    aliases: Tuple[str, ...] = field(default_factory=tuple)
    category: str = "General"
    source: str = "builtin"
    builtin: bool = False
    path: str = ""


def project_skills_dir() -> str:
    """The workspace-local skill folder, stable for the life of a launch."""
    root = os.environ.get("DIZEL_PROJECT_DIR") or os.getcwd()
    return os.path.join(root, ".agent", "skills")


def _split_aliases(raw: str) -> Tuple[str, ...]:
    value = raw.strip().strip("[]")
    items = re.split(r"[,;]", value)
    cleaned = []
    for item in items:
        alias = _slugify(item.strip().strip("'\""))
        if alias and alias not in cleaned:
            cleaned.append(alias)
    return tuple(cleaned)


def _slugify(value: str) -> str:
    return _SLUG.sub("-", value.strip().lower()).strip("-")


def _parse(path: str, source: str, fallback_name: Optional[str] = None) -> Optional[Skill]:
    try:
        if os.path.getsize(path) > 200_000:
            return None
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return None

    meta = {}
    match = _FRONTMATTER.match(text)
    if match:
        body = text[match.end():]
        for line in match.group(1).splitlines():
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            meta[key.strip().lower()] = value.strip()
    else:
        body = text.lstrip("\ufeff")

    stem = os.path.splitext(os.path.basename(path))[0]
    if fallback_name and stem.lower() == "skill":
        stem = fallback_name
    name = _slugify(meta.get("name") or stem)
    if not name:
        return None

    body = body.strip()
    if not body:
        return None

    description = meta.get("description") or ""
    if not description:
        first = next((line.strip() for line in body.splitlines() if line.strip()), "")
        description = first[:140]

    default_category = "Local" if source == "project" else "General"
    category = " ".join((meta.get("category") or default_category).split())[:40] or default_category

    return Skill(
        name=name,
        title=meta.get("title") or name.replace("-", " ").title(),
        description=description,
        body=body,
        aliases=_split_aliases(meta.get("aliases", "")),
        category=category,
        source=source,
        builtin=source == "builtin",
        path=path,
    )


def _scan(directory: str, source: str, folders_only: bool = False) -> List[Skill]:
    if not directory or not os.path.isdir(directory):
        return []
    found = []
    for entry in sorted(os.listdir(directory)):
        path = os.path.join(directory, entry)
        if os.path.isdir(path):
            skill_file = os.path.join(path, "SKILL.md")
            if os.path.isfile(skill_file):
                skill = _parse(skill_file, source, fallback_name=entry)
                if skill:
                    found.append(skill)
            continue
        if folders_only or not entry.lower().endswith(".md"):
            continue
        skill = _parse(path, source)
        if skill:
            found.append(skill)
    return found


def list_skills(
    user_dir: Optional[str] = None,
    builtin_dir: Optional[str] = None,
    project_dir: Optional[str] = None,
) -> List[Skill]:
    """Built-ins, user skills, then project skills; later sources win."""
    builtin_path = BUILTIN_DIR if builtin_dir is None else builtin_dir
    user_path = USER_DIR if user_dir is None else user_dir
    project_path = project_skills_dir() if project_dir is None else project_dir

    catalog: dict[str, Skill] = {}
    for skill in _scan(builtin_path, "builtin"):
        catalog[skill.name] = skill
    for skill in _scan(user_path, "user"):
        catalog[skill.name] = skill
    for skill in _scan(project_path, "project", folders_only=True):
        catalog[skill.name] = skill
    return sorted(catalog.values(), key=lambda item: item.name)


def get_skill(
    name: str,
    user_dir: Optional[str] = None,
    builtin_dir: Optional[str] = None,
    project_dir: Optional[str] = None,
) -> Optional[Skill]:
    """Match a skill by name or alias, case-insensitively."""
    needle = _slugify(name)
    if not needle:
        return None
    skills = list_skills(user_dir=user_dir, builtin_dir=builtin_dir, project_dir=project_dir)
    for skill in skills:
        if skill.name == needle:
            return skill
    for skill in skills:
        if needle in skill.aliases:
            return skill
    return None


def group_skills(skills: List[Skill]) -> List[Tuple[str, List[Skill]]]:
    """Skills bucketed by category, in the shared reading order."""
    rank = {name: index for index, name in enumerate(CATEGORY_ORDER)}
    buckets: dict[str, List[Skill]] = {}
    for skill in skills:
        buckets.setdefault(skill.category, []).append(skill)
    ordered = sorted(buckets, key=lambda name: (rank.get(name, len(CATEGORY_ORDER)), name))
    return [(name, buckets[name]) for name in ordered]


def expand_skill(skill: Skill, args: str = "") -> str:
    """Turn a skill body into the prompt placed in the composer."""
    text = (args or "").strip()
    body = skill.body.strip()
    if INPUT_TOKEN in body:
        return body.replace(INPUT_TOKEN, text or EMPTY_INPUT).strip()
    if text:
        return f"{body}\n\nInput:\n{text}"
    return body


