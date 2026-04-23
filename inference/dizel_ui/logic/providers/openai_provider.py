"""
logic/providers/openai_provider.py — OpenAI API integration.

Uses the official `openai` SDK for chat completions and model listing.
"""

from typing import Dict, Generator, List

from .base import BaseProvider, ProviderInfo, ProviderModel


class OpenAIProvider(BaseProvider):
    info = ProviderInfo(
        name="OpenAI",
        slug="openai",
        avatar_file="chatgpt.png",
        requires_key=True,
        description="GPT-4o, o3, and more from OpenAI",
        base_url="https://api.openai.com/v1",
    )

    def validate(self, key: str = "", **kwargs) -> bool:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")

        if not key:
            raise ValueError("API key is required for OpenAI.")

        try:
            client = OpenAI(api_key=key)
            client.models.list()
            return True
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "Incorrect API key" in error_msg:
                raise ValueError("Invalid OpenAI API key.")
            raise ConnectionError(f"OpenAI connection error: {e}")

    def list_models(self, key: str = "", **kwargs) -> List[ProviderModel]:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")

        client = OpenAI(api_key=key)
        try:
            response = client.models.list()
        except Exception as e:
            raise ConnectionError(f"Failed to list OpenAI models: {e}")

        chat_prefixes = ("gpt-", "o1", "o3", "o4", "chatgpt-")
        models = []
        for m in response.data:
            mid = m.id
            if any(mid.startswith(p) for p in chat_prefixes):
                models.append(ProviderModel(
                    id=mid,
                    name=mid,
                    context_length=0,
                ))

        models.sort(key=lambda x: x.id)
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
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")

        client = OpenAI(api_key=key)
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as e:
            raise RuntimeError(f"OpenAI generation error: {e}")
