"""
logic/providers/anthropic_provider.py — Anthropic Claude API integration.

Uses the official `anthropic` SDK. Model list is hardcoded since
Anthropic doesn't provide a list endpoint.
"""

from typing import Dict, Generator, List

from .base import BaseProvider, ProviderInfo, ProviderModel

_CLAUDE_MODELS = [
    ProviderModel(id="claude-sonnet-4-20250514", name="Claude Sonnet 4", context_length=200000),
    ProviderModel(id="claude-opus-4-20250514", name="Claude Opus 4", context_length=200000),
    ProviderModel(id="claude-3-5-haiku-20241022", name="Claude 3.5 Haiku", context_length=200000),
    ProviderModel(id="claude-3-5-sonnet-20241022", name="Claude 3.5 Sonnet", context_length=200000),
]


class AnthropicProvider(BaseProvider):
    info = ProviderInfo(
        name="Anthropic",
        slug="anthropic",
        avatar_file="claude.png",
        requires_key=True,
        description="Claude AI from Anthropic",
    )

    def validate(self, key: str = "", **kwargs) -> bool:
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")

        if not key:
            raise ValueError("API key is required for Anthropic.")

        try:
            client = anthropic.Anthropic(api_key=key)
            client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True
        except anthropic.AuthenticationError:
            raise ValueError("Invalid Anthropic API key.")
        except Exception as e:
            raise ConnectionError(f"Anthropic connection error: {e}")

    def list_models(self, key: str = "", **kwargs) -> List[ProviderModel]:
        return list(_CLAUDE_MODELS)

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        key: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> Generator[str, None, None]:
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")

        client = anthropic.Anthropic(api_key=key)

        # Separate system message from conversation
        system_msg = ""
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                chat_messages.append(m)

        try:
            with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                system=system_msg if system_msg else anthropic.NOT_GIVEN,
                messages=chat_messages,
                temperature=temperature,
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as e:
            raise RuntimeError(f"Anthropic generation error: {e}")
