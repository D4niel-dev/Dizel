"""
logic/providers/google_provider.py — Google Gemini API integration.

Uses the `google-genai` SDK for model listing and streaming.
"""

from typing import Dict, Generator, List

from .base import BaseProvider, ProviderInfo, ProviderModel

_GEMINI_MODELS = [
    ProviderModel(id="gemini-2.5-flash", name="Gemini 2.5 Flash", context_length=1048576),
    ProviderModel(id="gemini-2.5-pro", name="Gemini 2.5 Pro", context_length=1048576),
    ProviderModel(id="gemini-2.0-flash", name="Gemini 2.0 Flash", context_length=1048576),
    ProviderModel(id="gemini-1.5-pro", name="Gemini 1.5 Pro", context_length=2097152),
]


class GoogleProvider(BaseProvider):
    info = ProviderInfo(
        name="Google (Gemini)",
        slug="google",
        avatar_file="gemini.png",
        requires_key=True,
        description="Gemini AI from Google",
    )

    def validate(self, key: str = "", **kwargs) -> bool:
        try:
            from google import genai
        except ImportError:
            raise ImportError("pip install google-genai")

        if not key:
            raise ValueError("API key is required for Google Gemini.")

        try:
            client = genai.Client(api_key=key)
            client.models.list()
            return True
        except Exception as e:
            error_msg = str(e)
            if "API_KEY_INVALID" in error_msg or "401" in error_msg:
                raise ValueError("Invalid Google Gemini API key.")
            raise ConnectionError(f"Google Gemini connection error: {e}")

    def list_models(self, key: str = "", **kwargs) -> List[ProviderModel]:
        # Return curated list — Google's full list includes non-chat models
        return list(_GEMINI_MODELS)

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
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("pip install google-genai")

        client = genai.Client(api_key=key)

        # Convert OpenAI message format to Gemini format
        system_instruction = None
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(text=msg["content"])],
                ))
            elif msg["role"] == "assistant":
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part(text=msg["content"])],
                ))

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system_instruction:
            config.system_instruction = system_instruction

        try:
            response = client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            raise RuntimeError(f"Gemini generation error: {e}")
