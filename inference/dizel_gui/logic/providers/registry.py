"""
logic/providers/registry.py — Central provider registry.

Maps provider slugs to their implementation classes.
"""

from typing import Dict, Optional, Type

from .base import BaseProvider, ProviderInfo


def _build_registry() -> Dict[str, Type[BaseProvider]]:
    """Lazy-build the registry to avoid import-time side effects."""
    from .ollama_provider import OllamaProvider
    from .openai_provider import OpenAIProvider
    from .anthropic_provider import AnthropicProvider
    from .google_provider import GoogleProvider
    from .generic_openai_provider import (
        GroqProvider, MistralProvider, XAIProvider,
        AI21Provider, AzureProvider, CohereProvider, MetaProvider,
    )

    return {
        "ollama":    OllamaProvider,
        "openai":    OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google":    GoogleProvider,
        "groq":      GroqProvider,
        "mistral":   MistralProvider,
        "xai":       XAIProvider,
        "ai21":      AI21Provider,
        "azure":     AzureProvider,
        "cohere":    CohereProvider,
        "meta":      MetaProvider,
    }


_registry: Optional[Dict[str, Type[BaseProvider]]] = None


class ProviderRegistry:
    """Registry of all available AI providers."""

    @staticmethod
    def all() -> Dict[str, Type[BaseProvider]]:
        global _registry
        if _registry is None:
            _registry = _build_registry()
        return _registry

    @staticmethod
    def get(slug: str) -> Optional[Type[BaseProvider]]:
        return ProviderRegistry.all().get(slug)

    @staticmethod
    def slugs() -> list:
        return list(ProviderRegistry.all().keys())

    @staticmethod
    def infos() -> list:
        """Return ProviderInfo for all registered providers."""
        return [cls.info for cls in ProviderRegistry.all().values()]


def get_provider(slug: str, **kwargs) -> BaseProvider:
    """
    Instantiate a provider by slug.
    Pass kwargs like ollama_url for Ollama.
    """
    cls = ProviderRegistry.get(slug)
    if cls is None:
        raise ValueError(f"Unknown provider: '{slug}'. Available: {ProviderRegistry.slugs()}")

    # Pass relevant kwargs to constructors that accept them
    if slug == "ollama":
        return cls(base_url=kwargs.get("ollama_url", "http://localhost:11434"))
    return cls()
