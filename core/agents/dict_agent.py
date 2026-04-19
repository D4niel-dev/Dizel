"""
dict_agent.py — Image processing agent.

Dict uses ViT-GPT2 to generate text descriptions from images.
The vision model is loaded lazily on first use.
"""

import os
import traceback

from core.agents.base_agent import BaseAgent, AgentResult


# Lazy-loaded model reference
_captioner = None


def _load_captioner():
    """Load ViT-GPT2 on first use. ~600MB download on first run."""
    global _captioner
    if _captioner is not None:
        return _captioner

    try:
        from transformers import pipeline
        _captioner = pipeline(
            "image-to-text",
            model="nlpconnect/vit-gpt2-image-captioning",
            device=-1,   # CPU by default; change to 0 for GPU
        )
        return _captioner
    except ImportError:
        raise ImportError(
            "transformers and Pillow are required for image captioning. "
            "Install with: pip install transformers Pillow"
        )


class DictAgent(BaseAgent):
    """Generates text descriptions from images for Dizel."""

    @property
    def name(self) -> str:
        return "Dict"

    def process(self, input_path: str, **kwargs) -> AgentResult:
        file_name = os.path.basename(input_path)
        ext = os.path.splitext(file_name)[1].lower().lstrip(".")

        try:
            from PIL import Image
        except ImportError:
            return AgentResult(
                source="Dict",
                file_name=file_name,
                error="Pillow is required for image processing. pip install Pillow",
            )

        # Validate and get image info
        try:
            img = Image.open(input_path)
            width, height = img.size
            img_format = img.format or ext.upper()
        except Exception as e:
            return AgentResult(
                source="Dict",
                file_name=file_name,
                error=f"Cannot open image: {e}",
            )

        # Generate caption
        try:
            captioner = _load_captioner()
            results = captioner(input_path, max_new_tokens=100)
            caption = results[0].get("generated_text", "No description generated.")
        except ImportError as e:
            return AgentResult(
                source="Dict",
                file_name=file_name,
                error=str(e),
            )
        except Exception as e:
            return AgentResult(
                source="Dict",
                file_name=file_name,
                error=f"Captioning failed: {e}\n{traceback.format_exc()}",
            )

        details = [
            f"Resolution: {width}x{height}",
            f"Format: {img_format}",
        ]

        return AgentResult(
            source="Dict",
            file_name=file_name,
            file_type=f"image/{ext}",
            description=caption,
            details=details,
            notes="Image processed with ViT-GPT2 captioning model.",
        )
