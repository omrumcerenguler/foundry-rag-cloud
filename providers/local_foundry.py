"""Offline Foundry Local adapters with deferred SDK loading.

All inference calls target the on-device Foundry Local runtime. The configured
``model_path`` is passed directly to the local client for offline model
resolution; this module does not create outbound network connections.
"""

import importlib
from typing import Any

from core.ports import BaseChatProvider, BaseEmbeddingProvider


def _client(model_path: str | None = None) -> Any:
    """Load the optional SDK only when a local provider is actually invoked."""
    try:
        try:
            module = importlib.import_module("foundry_local_sdk")
        except ImportError:
            module = importlib.import_module("foundry_local")
    except ImportError as exc:
        raise RuntimeError(
            "Foundry Local is required for LOCAL mode; install foundry-local-sdk."
        ) from exc
    for name in ("FoundryLocalClient", "Client", "FoundryLocal"):
        client_type = getattr(module, name, None)
        if client_type is not None:
            return client_type(model_path=model_path) if model_path else client_type()
    raise RuntimeError("foundry-local-sdk does not expose a supported client")


def _embedding_result(response: Any) -> list[float]:
    """Extract an embedding from common SDK response shapes."""
    if hasattr(response, "data"):
        response = response.data[0]
    if isinstance(response, dict):
        response = response.get("embedding", response.get("data", response))
    if hasattr(response, "embedding"):
        response = response.embedding
    if not isinstance(response, list):
        raise RuntimeError("Foundry embedding response has an unsupported shape")
    return [float(value) for value in response]


class LocalFoundryEmbeddingProvider(BaseEmbeddingProvider):
    """Generate embeddings through the optional Foundry Local SDK."""

    def __init__(
        self,
        model_name: str = "qwen3-embedding-0.6b",
        dimension: int = 1024,
        model_path: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._dimension = dimension
        self._model_path = model_path

    @property
    def dimension(self) -> int:
        """Return the configured embedding dimension."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return the Foundry embedding model name."""
        return self._model_name

    def embed_text(self, text: str) -> list[float]:
        """Embed one text using the SDK's OpenAI-compatible embeddings API."""
        if not text.strip():
            raise ValueError("text must not be empty")
        try:
            client = _client(self._model_path)
            response = client.embeddings.create(model=self._model_name, input=text)
            embedding = _embedding_result(response)
        except Exception as exc:
            raise RuntimeError(f"Foundry embedding failed: {exc}") from exc
        if len(embedding) != self._dimension:
            raise RuntimeError(
                f"Expected {self._dimension} dimensions, received {len(embedding)}"
            )
        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch while preserving input order."""
        if not texts:
            return []
        return [self.embed_text(text) for text in texts]


class LocalFoundryChatProvider(BaseChatProvider):
    """Generate chat responses through Foundry Local."""

    def __init__(self, model_name: str = "Phi-3.5-mini", model_path: str | None = None) -> None:
        self._model_name = model_name
        self._model_path = model_path

    @property
    def model_name(self) -> str:
        """Return the Foundry chat model name."""
        return self._model_name

    def generate_response(
        self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float
    ) -> str:
        """Generate one response with bounded token and temperature settings."""
        try:
            client = _client(self._model_path)
            response = client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = response.choices[0].message.content
        except Exception as exc:
            raise RuntimeError(f"Foundry chat completion failed: {exc}") from exc
        if not content:
            raise RuntimeError("Foundry chat response was empty")
        return str(content)
