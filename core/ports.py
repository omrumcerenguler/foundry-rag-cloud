"""Dependency inversion ports for providers and vector stores."""

from abc import ABC, abstractmethod

from .models import DocumentChunk, SearchResult


class BaseEmbeddingProvider(ABC):
    """Port implemented by text embedding providers."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Return an embedding for one text."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings in the same order as ``texts``."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the provider embedding dimension."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the deployed model identifier."""


class BaseChatProvider(ABC):
    """Port implemented by chat-completion providers."""

    @abstractmethod
    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate a response from system and user messages."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the deployed chat model identifier."""


class BaseVectorStore(ABC):
    """Port for persistence and similarity search over document chunks."""

    @abstractmethod
    def save_document(
        self, source_id: str, chunk_index: int, text: str, embedding: list[float]
    ) -> None:
        """Persist or replace one document chunk."""

    @abstractmethod
    def search_similar(
        self, query_embedding: list[float], top_k: int
    ) -> list[SearchResult]:
        """Return the highest-scoring chunks for an embedding."""

    @abstractmethod
    def clear_all(self) -> None:
        """Remove all stored document chunks."""

    def replace_all_atomic(
        self, _chunks: list[DocumentChunk], _metadata: dict[str, object]
    ) -> None:
        """Replace the complete index and metadata atomically."""
        raise NotImplementedError

    def get_index_metadata(self) -> dict[str, object] | None:
        """Return index metadata when the store supports metadata queries."""
        return None

    def check_health(self) -> bool:
        """Verify that the store can answer health queries."""
        return True

    def close(self) -> None:
        """Release store resources."""
