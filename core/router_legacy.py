"""
router.py — Detect input type and dispatch to the correct agent.
"""

import os
from enum import Enum, auto
from typing import Optional

from core.agents.base_agent import AgentResult


class InputType(Enum):
    TEXT = auto()
    IMAGE = auto()
    FILE = auto()
    UNSUPPORTED = auto()


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp",
}

FILE_EXTENSIONS = {
    # Documents
    ".pdf", ".docx", ".doc",
    # Text / Markdown
    ".txt", ".md",
    # Data
    ".json", ".jsonl", ".csv", ".tsv", ".xlsx", ".xls",
    # Code
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".html", ".css", ".scss",
    ".c", ".cpp", ".h", ".hpp",
    ".java", ".go", ".rs", ".rb",
    ".sh", ".bat", ".ps1",
    ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".xml", ".sql",
}

# Safety limits
MAX_IMAGE_SIZE_MB = 5
MAX_FILE_SIZE_MB = 10
MAX_IMAGES_PER_REQUEST = 3


def detect_input_type(file_path: str) -> InputType:
    """Classify a file path by extension."""
    if not file_path or not os.path.isfile(file_path):
        return InputType.UNSUPPORTED

    ext = os.path.splitext(file_path)[1].lower()

    if ext in IMAGE_EXTENSIONS:
        return InputType.IMAGE
    if ext in FILE_EXTENSIONS:
        return InputType.FILE
    return InputType.UNSUPPORTED


def validate_file(file_path: str, input_type: InputType) -> Optional[str]:
    """Return an error string if the file fails validation, else None."""
    if not os.path.isfile(file_path):
        return f"File not found: {file_path}"

    size_mb = os.path.getsize(file_path) / (1024 * 1024)

    if input_type == InputType.IMAGE and size_mb > MAX_IMAGE_SIZE_MB:
        return f"Image too large ({size_mb:.1f} MB). Limit is {MAX_IMAGE_SIZE_MB} MB."

    if input_type == InputType.FILE and size_mb > MAX_FILE_SIZE_MB:
        return f"File too large ({size_mb:.1f} MB). Limit is {MAX_FILE_SIZE_MB} MB."

    return None


def route_input(file_path: str) -> AgentResult:
    """Detect type → validate → dispatch to Dict or Lily → return result."""
    input_type = detect_input_type(file_path)

    if input_type == InputType.UNSUPPORTED:
        ext = os.path.splitext(file_path)[1] if file_path else "unknown"
        return AgentResult(
            source="Router",
            file_name=os.path.basename(file_path) if file_path else "",
            error=f"Unsupported file type: {ext}",
        )

    error = validate_file(file_path, input_type)
    if error:
        return AgentResult(
            source="Router",
            file_name=os.path.basename(file_path),
            error=error,
        )

    if input_type == InputType.IMAGE:
        from core.agents.dict_agent import DictAgent
        return DictAgent().process(file_path)

    if input_type == InputType.FILE:
        from core.agents.lily_agent import LilyAgent
        return LilyAgent().process(file_path)

    return AgentResult(source="Router", error="Unknown routing error.")


def route_multiple(file_paths: list) -> list:
    """Route multiple attachments. Enforces image count limit."""
    results = []
    image_count = 0

    for path in file_paths:
        inp_type = detect_input_type(path)
        if inp_type == InputType.IMAGE:
            image_count += 1
            if image_count > MAX_IMAGES_PER_REQUEST:
                results.append(AgentResult(
                    source="Router",
                    file_name=os.path.basename(path),
                    error=f"Too many images. Limit is {MAX_IMAGES_PER_REQUEST} per request.",
                ))
                continue
        results.append(route_input(path))

    return results


# ---------------------------------------------------------------------------
# Tool-aware request routing
# ---------------------------------------------------------------------------
def route_request(state) -> None:
    """
    Execute active tool pipelines in order based on ToolState.

    Pipeline order:  files → web search
    Deep Think is NOT handled here (it's a generation concern).

    Mutates `state` in place with populated context fields.
    """
    # Step 1: Parse files (only when toggle is ON + files attached)
    if state.parse_files_enabled and state.uploaded_files:
        state.processing_status = "parsing_files"
        results = route_multiple(state.uploaded_files)
        from core.prompt_builder import build_context
        state.file_context = build_context(results, "")

    # Step 2: Web search
    if state.web_search_enabled and state.user_input.strip():
        state.processing_status = "searching_web"
        try:
            from core.tools.web_search import search_web
            state.web_results = search_web(state.user_input)
        except Exception as e:
            print(f"[router] Web search failed: {e}")
            state.web_results = ""

    state.processing_status = "done"
