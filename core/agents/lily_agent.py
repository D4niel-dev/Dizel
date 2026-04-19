"""
lily_agent.py — File extraction agent.

Lily reads documents, code, and data files, extracts their text content,
cleans it, and returns a structured AgentResult for Dizel.
"""

import os
import traceback

from core.agents.base_agent import BaseAgent, AgentResult
from core.tools.file_extractors import extract_text
from core.tools.text_cleaner import clean_text
from core.tools.summarizer import summarize_if_needed


class LilyAgent(BaseAgent):
    """Extracts and processes file content into clean text for Dizel."""

    @property
    def name(self) -> str:
        return "Lily"

    def process(self, input_path: str, **kwargs) -> AgentResult:
        file_name = os.path.basename(input_path)

        try:
            raw_text, file_type, meta = extract_text(input_path)
        except ImportError as e:
            return AgentResult(
                source="Lily",
                file_name=file_name,
                error=f"Missing dependency: {e}",
            )
        except ValueError as e:
            return AgentResult(
                source="Lily",
                file_name=file_name,
                error=str(e),
            )
        except Exception as e:
            return AgentResult(
                source="Lily",
                file_name=file_name,
                error=f"Extraction failed: {e}\n{traceback.format_exc()}",
            )

        # Clean the extracted text
        cleaned = clean_text(raw_text)

        if not cleaned:
            return AgentResult(
                source="Lily",
                file_name=file_name,
                file_type=file_type,
                description="File appears to be empty or contains no extractable text.",
                notes="No content could be extracted.",
            )

        # Summarize/truncate if too long
        processed, trunc_note = summarize_if_needed(cleaned)

        # Build details from metadata
        details = []
        if "pages" in meta:
            details.append(f"{meta['pages']} pages")
        if "paragraphs" in meta:
            details.append(f"{meta['paragraphs']} paragraphs")
        if "lines" in meta:
            details.append(f"{meta['lines']} lines")
        if "total_rows" in meta:
            details.append(f"{meta['total_rows']} rows, {meta.get('columns', '?')} columns")
        if "sheets" in meta:
            details.append(f"{meta['sheets']} sheets")
        if "total_entries" in meta:
            details.append(f"{meta['total_entries']} entries")
        if "language" in meta:
            details.append(f"Language: {meta['language']}")

        # Build a brief description
        char_count = len(cleaned)
        description = (
            f"Extracted {char_count:,} characters of text from {file_type} file."
        )

        return AgentResult(
            source="Lily",
            file_name=file_name,
            file_type=file_type,
            description=description,
            details=details,
            raw_text=processed,
            notes=trunc_note,
        )
