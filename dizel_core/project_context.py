"""Project rules and memory composed into the model's system prompt."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from . import memory_store

RULES_FILES = ("DIZEL.md", "AGENTS.md")
MAX_CHARS = 12000
MEMORY_MAX_CHARS = 6000

PathLike = Union[str, Path, None]


@dataclass
class ProjectContext:
    root: str
    rules: str = ""
    sources: List[dict] = field(default_factory=list)
    memory: List[dict] = field(default_factory=list)
    memory_text: str = ""
    truncated: bool = False


def _default_root() -> Path:
    try:
        from . import tools

        return tools._project_root()
    except Exception:
        return Path.cwd()


def load(project_root: PathLike = None) -> ProjectContext:
    root = Path(project_root) if project_root else _default_root()
    try:
        root = root.expanduser().resolve()
    except OSError:
        pass

    sections: List[str] = []
    sources: List[dict] = []
    for name in RULES_FILES:
        path = root / name
        try:
            text = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            continue
        if not text:
            continue
        sources.append({"name": name, "path": str(path), "chars": len(text)})
        sections.append(f"## {name}\n{text}")

    rules = "\n\n".join(sections)
    truncated = False
    if len(rules) > MAX_CHARS:
        rules = rules[:MAX_CHARS].rstrip() + "\n\n[project rules truncated]"
        truncated = True

    rows = memory_store.entries(root)
    memory_text = "\n".join(f"- {row.get('text', '')}" for row in rows)
    if len(memory_text) > MEMORY_MAX_CHARS:
        memory_text = (
            memory_text[:MEMORY_MAX_CHARS].rstrip() + "\n- …[memory truncated]"
        )

    return ProjectContext(
        root=str(root),
        rules=rules,
        sources=sources,
        memory=rows,
        memory_text=memory_text,
        truncated=truncated,
    )


def system_blocks(context: ProjectContext) -> str:
    parts: List[str] = []
    if context.rules:
        parts.append("Project rules (from DIZEL.md / AGENTS.md):\n" + context.rules)
    if context.memory_text:
        parts.append(
            "Project memory (durable facts for this project):\n" + context.memory_text
        )
    return "\n\n".join(parts)


def rules_summary(context: ProjectContext) -> str:
    if not context.sources:
        return "No project rules files found (DIZEL.md, AGENTS.md)."
    lines = [f"{source['name']} — {source['chars']:,} chars" for source in context.sources]
    if context.truncated:
        lines.append(f"Combined rules truncated at {MAX_CHARS:,} characters.")
    return "\n".join(lines)
