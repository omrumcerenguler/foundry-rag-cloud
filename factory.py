"""Factory for assembling a complete RAG service from environment settings."""

from config import Settings
from core.ports import BaseChatProvider, BaseEmbeddingProvider
from core.service import RAGService
from providers.azure_openai import AzureOpenAIChatProvider, AzureOpenAIEmbeddingProvider
from providers.local_foundry import (
    LocalFoundryChatProvider,
    LocalFoundryEmbeddingProvider,
)
from stores.sqlite_store import SQLiteVectorStore


def get_rag_service(settings: Settings | None = None) -> RAGService:
    """Build a RAG service for ``LOCAL`` or ``AZURE_CLOUD`` mode."""
    resolved = settings or Settings.from_env()
    if resolved.rag_mode == "LOCAL":
        embedding_provider: BaseEmbeddingProvider = LocalFoundryEmbeddingProvider(
            resolved.local_embedding_model, model_path=resolved.foundry_model_path
        )
        chat_provider: BaseChatProvider = LocalFoundryChatProvider(
            resolved.local_chat_model, model_path=resolved.foundry_model_path
        )
    else:
        if not resolved.azure_endpoint or not resolved.azure_api_key:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY are required for cloud mode"
            )
        embedding_provider = AzureOpenAIEmbeddingProvider(
            resolved.azure_endpoint,
            resolved.azure_api_key,
            resolved.azure_api_version,
            resolved.azure_embedding_deployment,
            resolved.azure_embedding_dimension,
        )
        chat_provider = AzureOpenAIChatProvider(
            resolved.azure_endpoint,
            resolved.azure_api_key,
            resolved.azure_api_version,
            resolved.azure_chat_deployment,
        )
    return RAGService(
        embedding_provider=embedding_provider,
        chat_provider=chat_provider,
        vector_store=SQLiteVectorStore(
            resolved.database_path,
            vector_dimension=embedding_provider.dimension,
            embedding_model=embedding_provider.model_name,
        ),
        mode=resolved.rag_mode,
        confidence_threshold=resolved.confidence_threshold,
    )
