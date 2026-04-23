"""
logic/providers/generic_openai_provider.py — OpenAI-compatible providers.

Many providers (Groq, Mistral, xAI, AI21, Azure, Cohere, Meta)
use the OpenAI SDK with a different base_url. This module creates
provider classes for all of them from a single template.
"""

from typing import Dict, Generator, List

from .base import BaseProvider, ProviderInfo, ProviderModel


class OpenAICompatibleProvider(BaseProvider):
    """
    Generic provider for any API that speaks the OpenAI chat completions protocol.
    Subclasses just set `info` and optionally override `list_models`.
    """

    def _get_client(self, key: str, **kwargs):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")

        extra = {}
        base = self.info.base_url

        # Azure needs special handling  
        if self.info.slug == "azure":
            try:
                from openai import AzureOpenAI
            except ImportError:
                raise ImportError("pip install openai")
            resource = kwargs.get("azure_resource", "")
            deployment = kwargs.get("azure_deployment", "")
            if not resource or not deployment:
                raise ValueError("Azure requires Resource Name and Deployment Name.")
            return AzureOpenAI(
                api_key=key,
                azure_endpoint=f"https://{resource}.openai.azure.com",
                azure_deployment=deployment,
                api_version="2024-12-01-preview",
            )

        return OpenAI(api_key=key, base_url=base)

    def validate(self, key: str = "", **kwargs) -> bool:
        if not key:
            raise ValueError(f"API key is required for {self.info.name}.")
        try:
            client = self._get_client(key, **kwargs)
            client.models.list()
            return True
        except ValueError:
            raise
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "Unauthorized" in error_msg:
                raise ValueError(f"Invalid {self.info.name} API key.")
            raise ConnectionError(f"{self.info.name} connection error: {e}")

    def list_models(self, key: str = "", **kwargs) -> List[ProviderModel]:
        try:
            client = self._get_client(key, **kwargs)
            response = client.models.list()
        except Exception as e:
            raise ConnectionError(f"Failed to list {self.info.name} models: {e}")

        models = []
        for m in response.data:
            models.append(ProviderModel(
                id=m.id,
                name=m.id,
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
        client = self._get_client(key, **kwargs)
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise RuntimeError(f"{self.info.name} generation error: {e}")


# ── Concrete Providers ──────────────────────────────────────────────────────

class GroqProvider(OpenAICompatibleProvider):
    info = ProviderInfo(
        name="Groq",
        slug="groq",
        avatar_file="Groq.png",
        requires_key=True,
        description="Ultra-fast inference from Groq",
        base_url="https://api.groq.com/openai/v1",
    )


class MistralProvider(OpenAICompatibleProvider):
    info = ProviderInfo(
        name="Mistral AI",
        slug="mistral",
        avatar_file="mistral-ai.png",
        requires_key=True,
        description="Open-weight models from Mistral AI",
        base_url="https://api.mistral.ai/v1",
    )


class XAIProvider(OpenAICompatibleProvider):
    info = ProviderInfo(
        name="xAI",
        slug="xai",
        avatar_file="xai.png",
        requires_key=True,
        description="Grok models from xAI",
        base_url="https://api.x.ai/v1",
    )


class AI21Provider(OpenAICompatibleProvider):
    info = ProviderInfo(
        name="AI21 Labs",
        slug="ai21",
        avatar_file="ai21-labs.png",
        requires_key=True,
        description="Jamba models from AI21 Labs",
        base_url="https://api.ai21.com/studio/v1",
    )


class AzureProvider(OpenAICompatibleProvider):
    info = ProviderInfo(
        name="Azure OpenAI",
        slug="azure",
        avatar_file="microsoft-azure-openaI.png",
        requires_key=True,
        description="OpenAI models via Microsoft Azure",
        base_url="",  # Built dynamically from resource name
    )


class CohereProvider(OpenAICompatibleProvider):
    info = ProviderInfo(
        name="Cohere",
        slug="cohere",
        avatar_file="cohere.png",
        requires_key=True,
        description="Command models from Cohere",
        base_url="https://api.cohere.com/compatibility/v1",
    )


class MetaProvider(OpenAICompatibleProvider):
    info = ProviderInfo(
        name="Meta (Llama)",
        slug="meta",
        avatar_file="meta.png",
        requires_key=True,
        description="Llama models via Meta's API",
        base_url="https://api.llama-api.com",
    )
