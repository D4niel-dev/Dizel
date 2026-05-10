"""
logic/providers/base.py — Abstract base class for all AI providers.

Every provider (OpenAI, Ollama, Anthropic, etc.) implements this interface
so the rest of Dizel can interact with any backend uniformly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional


@dataclass
class ProviderModel:
    """A single model available from a provider."""
    id: str           # "gpt-4o", "llama3:8b", "claude-sonnet-4-20250514"
    name: str         # Human display name: "GPT-4o", "Llama 3 8B"
    context_length: int = 0  # 0 = unknown


@dataclass
class ProviderInfo:
    """Static metadata about a provider."""
    name: str              # "OpenAI"
    slug: str              # "openai"
    avatar_file: str       # "chatgpt.png"
    requires_key: bool     # True (False for Ollama)
    description: str = ""  # "Cloud AI from OpenAI"
    base_url: str = ""     # Default API base URL


class BaseProvider(ABC):
    """Abstract interface every provider must implement."""

    info: ProviderInfo

    @abstractmethod
    def validate(self, key: str = "", **kwargs) -> bool:
        """
        Check whether the provider is reachable and the key is valid.
        For Ollama, checks if the server is running (key is ignored).
        Returns True on success.
        Raises ConnectionError or ValueError on failure with a clear message.
        """
        ...

    @abstractmethod
    def list_models(self, key: str = "", **kwargs) -> List[ProviderModel]:
        """
        Fetch available models from the provider.
        Returns a list of ProviderModel instances.
        """
        ...

    @abstractmethod
    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        key: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> Generator[str, None, None]:
        """
        Stream a chat completion. Yields token strings as they arrive.
        `messages` follows the OpenAI format:
            [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        """
        ...

    def get_avatar_path(self) -> str:
        """Return the filename for this provider's avatar."""
        return self.info.avatar_file
