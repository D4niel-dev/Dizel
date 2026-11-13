from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException

from dizel_core import skills as skills_core

router = APIRouter(prefix="/skills", tags=["skills"])


@router.get("/")
def list_skills() -> Dict[str, Any]:
    """Prompt templates available to every surface."""
    return {
        "skills": [
            {
                "name": skill.name,
                "title": skill.title,
                "description": skill.description,
                "aliases": list(skill.aliases),
                "category": skill.category,
                "source": skill.source,
                "builtin": skill.builtin,
            }
            for skill in skills_core.list_skills()
        ],
        "user_dir": skills_core.USER_DIR,
        "project_dir": skills_core.project_skills_dir(),
    }


@router.post("/expand")
def expand_skill(payload: Dict[str, Any] = Body(...)) -> Dict[str, str]:
    """Expand a skill into the prompt inserted into the composer."""
    name = str(payload.get("name") or "").strip()
    skill = skills_core.get_skill(name)
    if skill is None:
        raise HTTPException(status_code=404, detail=f"Unknown skill: {name or '(none)'}")
    return {
        "name": skill.name,
        "title": skill.title,
        "prompt": skills_core.expand_skill(skill, str(payload.get("args") or "")),
    }
