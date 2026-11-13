"""API providers for the model router.

Both `api/routes/providers.py` and `ChatManager` import `get_provider` from this
module. It did not exist, so every cloud provider failed at import time and the
manager quietly fell back to the local model — while the settings panel reported
"is Ollama running on localhost:11434?" for what was actually a ModuleNotFoundError.

Two shapes cover the catalogue:

* `OllamaProvider` speaks Ollama's own HTTP API (no key, local daemon).
* `OpenAICompatibleProvider` speaks ``/chat/completions`` with SSE, which is what
  OpenAI, Groq, Mistral, xAI, Google's compat endpoint and Cohere's compat
  endpoint all expose.
* `AnthropicProvider` is its own shape (``/v1/messages``, ``x-api-key``).

Anything not covered raises `ProviderError` with a reason, rather than pretending
to work.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional

import requests

DEFAULT_TIMEOUT = 20
STREAM_TIMEOUT = 300

# Model IDs are the only capability signal most provider APIs expose. These
# families are multimodal; anything else stays text-only.
_VISION_MODEL = re.compile(
    r"(?:"
    r"gpt-4o|gpt-4\.1|gpt-4-turbo|gpt-4-vision|"
    r"claude-3|claude-(?:sonnet|opus|haiku)-4|"
    r"gemini|grok-.*vision|pixtral|mistral-small-3\.1|aya-vision|"
    r"llava|bakllava|moondream|qwen[^ ]*vl|minicpm-v|internvl|"
    r"llama-?3\.2[^ ]*vision|llama-?4|phi-.*vision|gemma-?3|"
    r"vision|multimodal|omni"
    r")",
    re.IGNORECASE,
)


def model_supports_vision(provider: str, model: str) -> bool:
    """Best-effort detection for provider models; local backends report their own."""
    if not model:
        return False
    name = model.lower()
    if provider == "ollama":
        return bool(
            re.search(
                r"(llava|bakllava|moondream|vision|qwen.*vl|minicpm-v|internvl|gemma3)",
                name,
            )
        )
    return bool(_VISION_MODEL.search(name))


def split_data_url(value: str) -> Optional[tuple[str, str]]:
    """(media_type, base64) for image data URLs, or None."""
    if not isinstance(value, str) or not value.startswith("data:") or "," not in value:
        return None
    header, data = value.split(",", 1)
    if ";base64" not in header:
        return None
    media = header[5:].split(";", 1)[0] or "image/png"
    return media, data


class ProviderError(RuntimeError):
    """Something the user can act on: unreachable host, bad key, wrong model."""


@dataclass
class ModelInfo:
    id: str
    name: str


def _explain(exc: Exception, where: str) -> ProviderError:
    """Turn a requests exception into a sentence worth showing in the UI."""
    if isinstance(exc, requests.exceptions.ConnectTimeout):
        return ProviderError(f"Timed out connecting to {where}.")
    if isinstance(exc, requests.exceptions.ConnectionError):
        return ProviderError(
            f"Could not reach {where} — connection refused. Is the service running?"
        )
    if isinstance(exc, requests.exceptions.ReadTimeout):
        return ProviderError(f"{where} accepted the connection but did not answer in time.")
    return ProviderError(f"{where}: {exc}")


class BaseProvider:
    label = "Provider"

    def __init__(self) -> None:
        self._active_response = None
        self._stream_lock = threading.Lock()

    def cancel(self) -> None:
        """Close the active HTTP stream, if one is open.

        The stop button runs on a different thread from the one iterating the
        response. Closing the response here makes iter_lines() return at once
        instead of waiting for the provider's next chunk.
        """
        with self._stream_lock:
            response = self._active_response
        if response is not None:
            try:
                response.close()
            except Exception:
                pass

    def _track_response(self, response):
        with self._stream_lock:
            self._active_response = response
        return response

    def _release_response(self, response) -> None:
        with self._stream_lock:
            if self._active_response is response:
                self._active_response = None
        try:
            response.close()
        except Exception:
            pass

    def list_models(self, key: str = "") -> List[ModelInfo]:
        raise NotImplementedError

    def stream_events(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        key: str = "",
        temperature: float = 0.8,
        max_tokens: int = 512,
        **_: Any,
    ) -> Iterator[tuple[str, str]]:
        """Yield ``(kind, text)`` where kind is "thinking" or "content".

        Reasoning models produce thinking tokens before the answer. They are
        worth keeping — the GUI shows them in a collapsed block — but they must
        never be mixed into the reply text.
        """
        raise NotImplementedError

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        key: str = "",
        temperature: float = 0.8,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Content only, for callers that do not care about thinking."""
        for kind, text in self.stream_events(
            messages=messages,
            model=model,
            key=key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        ):
            if kind == "content":
                yield text


# ── Ollama ────────────────────────────────────────────────────────────────
class OllamaProvider(BaseProvider):
    label = "Ollama"

    def __init__(self, ollama_url: str = "http://localhost:11434") -> None:
        super().__init__()
        # Strip a trailing slash so joins never double up.
        self.base = (ollama_url or "http://localhost:11434").rstrip("/")

        # A daemon on loopback must never go through a proxy. With HTTP_PROXY set
        # in the environment — common in dev and corporate setups — requests
        # sends localhost:11434 to the proxy, which cannot reach it and answers
        # 502. That looked exactly like "Ollama is not running".
        self._session = requests.Session()
        self._session.trust_env = False

    def list_models(self, key: str = "") -> List[ModelInfo]:
        try:
            response = self._session.get(f"{self.base}/api/tags", timeout=DEFAULT_TIMEOUT)
        except Exception as exc:
            raise _explain(exc, self.base) from exc

        if response.status_code != 200:
            raise ProviderError(
                f"{self.base} answered {response.status_code}. "
                f"{response.text[:160].strip()}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderError(f"{self.base} did not return JSON.") from exc

        models = []
        for entry in payload.get("models", []):
            name = entry.get("name") or entry.get("model") or ""
            if name:
                models.append(ModelInfo(id=name, name=name))
        return models

    def stream_events(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        key: str = "",
        temperature: float = 0.8,
        max_tokens: int = 512,
        **_: Any,
    ) -> Iterator[tuple[str, str]]:
        if not model:
            raise ProviderError("No Ollama model selected. Pick one in Settings → Chat.")

        chat_messages = []
        for message in messages:
            item = {
                "role": message["role"],
                "content": message.get("content", ""),
            }
            images = []
            for value in message.get("images") or []:
                parsed = split_data_url(value)
                images.append(parsed[1] if parsed else value)
            if images:
                item["images"] = images
            chat_messages.append(item)

        body = {
            "model": model,
            "messages": chat_messages,
            "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        try:
            response = self._session.post(
                f"{self.base}/api/chat", json=body, stream=True, timeout=STREAM_TIMEOUT
            )
        except Exception as exc:
            raise _explain(exc, self.base) from exc

        if response.status_code == 404:
            raise ProviderError(f"Ollama has no model called '{model}'. Pull it first.")
        if response.status_code != 200:
            raise ProviderError(
                f"Ollama answered {response.status_code}. {response.text[:160].strip()}"
            )

        emitted = 0
        saw_thinking = False
        self._track_response(response)
        try:
            lines = response.iter_lines(decode_unicode=True)
            for line in lines:
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("error"):
                    raise ProviderError(str(event["error"]))

                message = event.get("message") or {}
                reasoning = message.get("thinking") or ""
                if reasoning:
                    saw_thinking = True
                    yield ("thinking", reasoning)
                chunk = message.get("content", "")
                if chunk:
                    emitted += 1
                    yield ("content", chunk)
                if event.get("done"):
                    break
        except requests.exceptions.RequestException as exc:
            raise ProviderError(f"Ollama stopped mid-reply: {exc}") from exc
        finally:
            self._release_response(response)

        # A reasoning model spends num_predict on thinking before it writes a
        # single visible token. Silence here looks like a broken provider.
        if emitted == 0 and not saw_thinking:
            raise ProviderError(f"'{model}' returned nothing.")


# ── OpenAI-compatible ─────────────────────────────────────────────────────
class OpenAICompatibleProvider(BaseProvider):
    """`/chat/completions` with SSE — the shape most vendors now ship."""

    def __init__(self, base_url: str, label: str = "Provider") -> None:
        super().__init__()
        self.base = base_url.rstrip("/")
        self.label = label

    def _headers(self, key: str) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def list_models(self, key: str = "") -> List[ModelInfo]:
        if not key:
            raise ProviderError(f"{self.label} needs an API key.")
        try:
            response = requests.get(
                f"{self.base}/models", headers=self._headers(key), timeout=DEFAULT_TIMEOUT
            )
        except Exception as exc:
            raise _explain(exc, self.base) from exc

        if response.status_code in (401, 403):
            raise ProviderError(f"{self.label} rejected the API key.")
        if response.status_code != 200:
            raise ProviderError(
                f"{self.label} answered {response.status_code}. "
                f"{response.text[:160].strip()}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderError(f"{self.label} did not return JSON.") from exc

        models = []
        for entry in payload.get("data", payload.get("models", [])):
            identifier = entry.get("id") or entry.get("name") or ""
            if identifier:
                models.append(ModelInfo(id=identifier, name=entry.get("name") or identifier))
        return models

    def stream_events(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        key: str = "",
        temperature: float = 0.8,
        max_tokens: int = 512,
        **_: Any,
    ) -> Iterator[tuple[str, str]]:
        if not key:
            raise ProviderError(f"{self.label} needs an API key.")
        if not model:
            raise ProviderError(f"No {self.label} model selected.")

        chat_messages = []
        for message in messages:
            images = message.get("images") or []
            if images:
                content: Any = [
                    {"type": "text", "text": message.get("content", "")},
                ]
                content.extend(
                    {"type": "image_url", "image_url": {"url": url}}
                    for url in images
                )
            else:
                content = message.get("content", "")
            chat_messages.append({"role": message["role"], "content": content})

        body = {
            "model": model,
            "messages": chat_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        try:
            response = requests.post(
                f"{self.base}/chat/completions",
                headers=self._headers(key),
                json=body,
                stream=True,
                timeout=STREAM_TIMEOUT,
            )
        except Exception as exc:
            raise _explain(exc, self.base) from exc

        if response.status_code in (401, 403):
            raise ProviderError(f"{self.label} rejected the API key.")
        if response.status_code == 404:
            raise ProviderError(f"{self.label} has no model called '{model}'.")
        if response.status_code != 200:
            raise ProviderError(
                f"{self.label} answered {response.status_code}. "
                f"{response.text[:160].strip()}"
            )

        self._track_response(response)
        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except ValueError:
                    continue
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    # Some vendors expose reasoning on the same delta.
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if reasoning:
                        yield ("thinking", reasoning)
                    chunk = delta.get("content") or ""
                    if chunk:
                        yield ("content", chunk)
        except requests.exceptions.RequestException as exc:
            raise ProviderError(f"{self.label} stopped mid-reply: {exc}") from exc
        finally:
            self._release_response(response)


# ── Anthropic ─────────────────────────────────────────────────────────────
class AnthropicProvider(BaseProvider):
    label = "Anthropic"

    def __init__(self, base_url: str = "https://api.anthropic.com/v1") -> None:
        super().__init__()
        self.base = base_url.rstrip("/")

    def _headers(self, key: str) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        }

    def list_models(self, key: str = "") -> List[ModelInfo]:
        if not key:
            raise ProviderError("Anthropic needs an API key.")
        try:
            response = requests.get(
                f"{self.base}/models", headers=self._headers(key), timeout=DEFAULT_TIMEOUT
            )
        except Exception as exc:
            raise _explain(exc, self.base) from exc

        if response.status_code in (401, 403):
            raise ProviderError("Anthropic rejected the API key.")
        if response.status_code != 200:
            raise ProviderError(f"Anthropic answered {response.status_code}.")

        payload = response.json()
        return [
            ModelInfo(id=entry["id"], name=entry.get("display_name") or entry["id"])
            for entry in payload.get("data", [])
            if entry.get("id")
        ]

    def stream_events(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        key: str = "",
        temperature: float = 0.8,
        max_tokens: int = 512,
        **_: Any,
    ) -> Iterator[tuple[str, str]]:
        if not key:
            raise ProviderError("Anthropic needs an API key.")

        # Anthropic takes the system prompt as a top-level field.
        system = "\n".join(
            str(m.get("content", "")) for m in messages if m["role"] == "system"
        )
        turns = []
        for message in messages:
            if message["role"] not in ("user", "assistant"):
                continue
            images = message.get("images") or []
            if images:
                blocks: List[Dict[str, Any]] = []
                for value in images:
                    parsed = split_data_url(value)
                    if not parsed:
                        continue
                    media, data = parsed
                    blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media,
                                "data": data,
                            },
                        }
                    )
                text = message.get("content", "")
                if text:
                    blocks.append({"type": "text", "text": text})
                turns.append({"role": message["role"], "content": blocks})
            else:
                turns.append(
                    {"role": message["role"], "content": message.get("content", "")}
                )
        body: Dict[str, Any] = {
            "model": model,
            "messages": turns,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if system:
            body["system"] = system

        try:
            response = requests.post(
                f"{self.base}/messages",
                headers=self._headers(key),
                json=body,
                stream=True,
                timeout=STREAM_TIMEOUT,
            )
        except Exception as exc:
            raise _explain(exc, self.base) from exc

        if response.status_code in (401, 403):
            raise ProviderError("Anthropic rejected the API key.")
        if response.status_code != 200:
            raise ProviderError(
                f"Anthropic answered {response.status_code}. {response.text[:160].strip()}"
            )

        self._track_response(response)
        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                try:
                    event = json.loads(line[5:].strip())
                except ValueError:
                    continue
                if event.get("type") == "content_block_delta":
                    delta = event.get("delta") or {}
                    text = delta.get("text") or ""
                    if not text:
                        continue
                    # "thinking_delta" is extended reasoning, not the answer.
                    kind = "thinking" if delta.get("type") == "thinking_delta" else "content"
                    yield (kind, text)
                elif event.get("type") == "message_stop":
                    break
        except requests.exceptions.RequestException as exc:
            raise ProviderError(f"Anthropic stopped mid-reply: {exc}") from exc
        finally:
            self._release_response(response)


# ── Registry ──────────────────────────────────────────────────────────────
_OPENAI_COMPATIBLE = {
    "openai": ("https://api.openai.com/v1", "OpenAI"),
    "groq": ("https://api.groq.com/openai/v1", "Groq"),
    "mistral": ("https://api.mistral.ai/v1", "Mistral AI"),
    "xai": ("https://api.x.ai/v1", "xAI"),
    "google": ("https://generativelanguage.googleapis.com/v1beta/openai", "Google"),
    "cohere": ("https://api.cohere.ai/compatibility/v1", "Cohere"),
    "ai21": ("https://api.ai21.com/studio/v1", "AI21 Labs"),
    "azure": ("https://api.openai.com/v1", "Azure OpenAI"),
    "meta": ("https://api.together.xyz/v1", "Meta (Llama)"),
}


def get_provider(slug: str, **kwargs: Any) -> BaseProvider:
    """Build the provider for `slug`. Raises ProviderError for anything unknown."""
    slug = (slug or "").lower()

    if slug == "ollama":
        return OllamaProvider(kwargs.get("ollama_url") or "http://localhost:11434")

    if slug == "anthropic":
        return AnthropicProvider()

    if slug == "custom":
        url = (kwargs.get("custom_url") or "").strip()
        if not url:
            raise ProviderError("Set a base URL for the custom provider first.")
        return OpenAICompatibleProvider(url, kwargs.get("custom_name") or "Custom Provider")

    if slug in _OPENAI_COMPATIBLE:
        base, label = _OPENAI_COMPATIBLE[slug]
        return OpenAICompatibleProvider(base, label)

    if slug == "local":
        raise ProviderError("The local model is not an API provider.")

    raise ProviderError(f"No provider implementation for '{slug}'.")
