import pytest

from core.models import RAGQueryRequest, SearchResult
from core.ports import BaseChatProvider, BaseEmbeddingProvider, BaseVectorStore
from core.service import MAX_CONTEXT_CHARACTERS, RAGService


class MockEmbedding(BaseEmbeddingProvider):
    @property
    def dimension(self) -> int:
        return 2

    @property
    def model_name(self) -> str:
        return "mock-embedding"

    def embed_text(self, text: str) -> list[float]:
        return [1.0, 0.0] if "known" in text else [0.0, 1.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


class MockChat(BaseChatProvider):
    @property
    def model_name(self) -> str:
        return "mock-chat"

    def generate_response(
        self,
        _system_prompt: str,
        user_prompt: str,
        _max_tokens: int,
        _temperature: float,
    ) -> str:
        assert "Sources:" in user_prompt
        return "Grounded answer [Source: doc1]"


class MockStore(BaseVectorStore):
    def save_document(
        self, _source_id: str, _chunk_index: int, _text: str, _embedding: list[float]
    ) -> None:
        pass

    def clear_all(self) -> None:
        pass

    def search_similar(
        self, query_embedding: list[float], _top_k: int
    ) -> list[SearchResult]:
        if query_embedding == [0.0, 1.0]:
            return [
                SearchResult(source_id="irrelevant", chunk_index=0, text="x", score=0.1)
            ]
        return [
            SearchResult(
                source_id="doc1", chunk_index=0, text="known context", score=0.9
            )
        ]


def test_grounding_citations_and_latency() -> None:
    service = RAGService(MockEmbedding(), MockChat(), MockStore())
    response = service.query(RAGQueryRequest(query="known question"))
    assert response.answer.startswith("Grounded")
    assert response.citations == ["doc1"]
    assert response.latency_seconds >= 0


def test_confidence_threshold_fallback() -> None:
    service = RAGService(
        MockEmbedding(), MockChat(), MockStore(), confidence_threshold=0.5
    )
    response = service.query(RAGQueryRequest(query="unrelated question"))
    assert "could not find" in response.answer
    assert response.sources == []
    assert response.citations == []


def test_empty_query_is_rejected() -> None:
    with pytest.raises(ValueError):
        RAGQueryRequest(query=" ")


def test_uncited_model_answer_falls_back() -> None:
    class UncitedChat(MockChat):
        def generate_response(
            self,
            _system_prompt: str,
            _user_prompt: str,
            _max_tokens: int,
            _temperature: float,
        ) -> str:
            return "unsupported answer"

    response = RAGService(MockEmbedding(), UncitedChat(), MockStore()).query(
        RAGQueryRequest(query="known question")
    )
    assert "could not find" in response.answer
    assert response.citations == []


def test_grounded_prompt_has_a_character_budget() -> None:
    result = SearchResult(
        source_id="doc1",
        chunk_index=0,
        text="x" * (MAX_CONTEXT_CHARACTERS * 2),
        score=0.9,
    )

    _, prompt = RAGService._build_grounded_prompt("question", [result])

    context = prompt.split("\n\n<user_query>", 1)[0]
    assert len(context.removeprefix("Sources:\n")) <= MAX_CONTEXT_CHARACTERS
