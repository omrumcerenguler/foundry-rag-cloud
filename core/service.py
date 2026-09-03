"""Application service coordinating retrieval and grounded generation."""

import html
import logging
import re
from time import perf_counter

from .models import RAGQueryRequest, RAGResponse, SearchResult
from .ports import BaseChatProvider, BaseEmbeddingProvider, BaseVectorStore

logger = logging.getLogger(__name__)


class RAGService:
    """Run the retrieval-augmented generation workflow behind stable ports."""

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        chat_provider: BaseChatProvider,
        vector_store: BaseVectorStore,
        mode: str = "LOCAL",
        confidence_threshold: float = 0.35,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.embedding_provider = embedding_provider
        self.chat_provider = chat_provider
        self.vector_store = vector_store
        self.mode = mode
        self.confidence_threshold = confidence_threshold

    def add_document(self, source_id: str, chunk_index: int, text: str) -> None:
        """Embed and persist one document chunk."""
        if not text.strip():
            raise ValueError("document text must not be empty")
        embedding = self.embedding_provider.embed_text(text)
        self.vector_store.save_document(source_id, chunk_index, text, embedding)

    def query(self, request: RAGQueryRequest) -> RAGResponse:
        """Retrieve context, generate a grounded answer, and record latency."""
        started = perf_counter()
        if not request.query.strip():
            raise ValueError("query must not be empty")
        retrieval_started = perf_counter()
        query_embedding = self.embedding_provider.embed_text(request.query)
        results = sorted(
            self.vector_store.search_similar(query_embedding, request.top_k),
            key=lambda result: result.score,
            reverse=True,
        )
        retrieval_ms = (perf_counter() - retrieval_started) * 1000
        threshold = (
            request.confidence_threshold
            if request.confidence_threshold is not None
            else self.confidence_threshold
        )
        if not results or results[0].score < threshold:
            answer = (
                "I could not find enough relevant information in the available sources."
            )
            results = []
            citations: list[str] = []
        else:
            generation_started = perf_counter()
            system_prompt, user_prompt = self._build_grounded_prompt(
                request.query, results
            )
            answer = self.chat_provider.generate_response(
                system_prompt, user_prompt, request.max_tokens, request.temperature
            )
            generation_ms = (perf_counter() - generation_started) * 1000
            citations = [
                result.source_id
                for result in results
                if re.search(rf"\b{re.escape(result.source_id)}\b", answer)
            ]
            if not citations:
                answer = "I could not find enough relevant information in the available sources."
                results = []
        if not results:
            generation_ms = 0.0
        total_latency_ms = (perf_counter() - started) * 1000
        logger.info(
            "RAG query latency",
            extra={
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_latency_ms": total_latency_ms,
            },
        )
        return RAGResponse(
            answer=answer,
            sources=results,
            citations=citations,
            latency_seconds=total_latency_ms / 1000,
            mode=self.mode,
        )

    @staticmethod
    def _build_grounded_prompt(
        query: str, results: list[SearchResult]
    ) -> tuple[str, str]:
        """Build a prompt that explicitly constrains the answer to retrieved context."""
        context = "\n\n".join(
            f"[Source: {result.source_id}, chunk: {result.chunk_index}]\n{result.text}"
            for result in results
        )
        return (
            "You answer only from the supplied sources. If the sources do not support "
            "the answer, say so clearly and cite the source identifiers.",
            f"Sources:\n{context}\n\n<user_query>\n{html.escape(query, quote=False)}\n"
            "</user_query>\nAnswer with source citations.",
        )
