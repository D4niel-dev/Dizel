"""
logic/providers/ollama_provider.py — Ollama local model integration.

Connects to a running Ollama instance at localhost:11434.
No API key required — just checks if the server is running.
"""

import json
from typing import Dict, Generator, List

from .base import BaseProvider, ProviderInfo, ProviderModel

_DEFAULT_URL = "http://localhost:11434"


class OllamaProvider(BaseProvider):
    info = ProviderInfo(
        name="Ollama",
        slug="ollama",
        avatar_file="ollama.png",
        requires_key=False,
        description="Run models locally with Ollama",
        base_url=_DEFAULT_URL,
    )

    def __init__(self, base_url: str = _DEFAULT_URL):
        self._base_url = base_url.rstrip("/")

    def validate(self, key: str = "", **kwargs) -> bool:
        import httpx
        url = kwargs.get("ollama_url", self._base_url)
        try:
            r = httpx.get(f"{url}/api/tags", timeout=5.0)
            r.raise_for_status()
            return True
        except httpx.ConnectError:
            raise ConnectionError(
                f"Ollama is not running at {url}.\n"
                "Start it with: ollama serve"
            )
        except Exception as e:
            raise ConnectionError(f"Cannot reach Ollama: {e}")

    def list_models(self, key: str = "", **kwargs) -> List[ProviderModel]:
        import httpx
        url = kwargs.get("ollama_url", self._base_url)
        try:
            r = httpx.get(f"{url}/api/tags", timeout=10.0)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            raise ConnectionError(f"Failed to list Ollama models: {e}")

        models = []
        for m in data.get("models", []):
            name = m.get("name", "unknown")
            # Extract a clean display name from the tag
            display = name.split(":")[0].replace("-", " ").title()
            models.append(ProviderModel(
                id=name,
                name=display,
                context_length=0,
            ))
        return models

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        key: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> Generator[str, None, None]:
        import httpx
        url = kwargs.get("ollama_url", self._base_url)
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            with httpx.stream(
                "POST",
                f"{url}/api/chat",
                json=payload,
                timeout=120.0,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield content
                        if chunk.get("done", False):
                            return
                    except json.JSONDecodeError:
                        continue
        except httpx.ConnectError:
            raise ConnectionError("Ollama connection lost. Is it still running?")
        except Exception as e:
            raise RuntimeError(f"Ollama generation error: {e}")
