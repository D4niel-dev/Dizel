"""Project rules and memory endpoints shared by both clients."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from dizel_core import memory_store, project_context, tools

router = APIRouter(prefix="/project", tags=["project"])


class MemoryText(BaseModel):
    text: str


class MemoryId(BaseModel):
    id: str


def _root():
    return tools._project_root()


@router.get("/memory")
def project_memory() -> Dict[str, Any]:
    context = project_context.load(_root())
    return {
        "project": context.root,
        "entries": context.memory,
        "rules": context.sources,
        "rules_summary": project_context.rules_summary(context),
        "rules_truncated": context.truncated,
    }


@router.post("/memory/add")
def add_memory(payload: MemoryText) -> Dict[str, Any]:
    try:
        return memory_store.add(payload.text, project_root=_root(), source="user")
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/memory/remove")
def remove_memory(payload: MemoryId) -> Dict[str, bool]:
    if not memory_store.remove(payload.id, project_root=_root()):
        raise HTTPException(status_code=404, detail="No such memory entry.")
    return {"removed": True}


@router.post("/memory/clear")
def clear_memory() -> Dict[str, int]:
    return {"removed": memory_store.clear(project_root=_root())}


@router.get("/git/diff")
def git_diff(mode: str = "review") -> Dict[str, Any]:
    """A capped, read-only diff plus the prompt and compact transcript line."""
    from dizel_core import git_tools

    if mode == "staged":
        change = git_tools.staged_diff()
        if not change.get("error"):
            change["prompt"] = git_tools.commit_prompt(change)
            change["display"] = (
                f"Draft a commit message · {len(change.get('files') or [])} staged files"
            )
    else:
        change = git_tools.working_diff()
        if not change.get("error"):
            change["prompt"] = git_tools.review_prompt(change)
            change["display"] = (
                f"Review the working-tree diff · {len(change.get('files') or [])} files"
            )
    return change
