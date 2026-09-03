import json
import sqlite3
import sys
from email.message import Message
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from urllib.error import HTTPError, URLError

import pytest

import evaluate
import main
import config
import providers.azure_openai as azure
from api import QueryRequest
from config import Settings
from core.models import DocumentChunk, RAGQueryRequest, RAGResponse, SearchResult
from core.ports import BaseChatProvider, BaseEmbeddingProvider, BaseVectorStore
from core.service import RAGService
from ingestion import ingest_directory
from providers.azure_openai import (
    AzureAuthenticationError,
    AzureAuthorizationError,
    AzureOpenAIChatProvider,
    AzureOpenAIEmbeddingProvider,
    AzureRateLimitError,
    AzureServerError,
)
from providers.local_foundry import (
    LocalFoundryChatProvider,
    LocalFoundryEmbeddingProvider,
)
from stores.sqlite_store import SQLiteVectorStore


@pytest.fixture(autouse=True)
def close_sqlite_stores(monkeypatch: pytest.MonkeyPatch):
    """Close every store created by a test, including failed-path cases."""
    stores: list[SQLiteVectorStore] = []
    store_type = SQLiteVectorStore
    original_init = store_type.__init__

    def track_store(store: SQLiteVectorStore, *args: object, **kwargs: object) -> None:
        cast(Any, original_init)(store, *args, **kwargs)
        stores.append(store)

    monkeypatch.setattr(store_type, "__init__", track_store)
    yield
    for store in stores:
        try:
            store.close()
        except sqlite3.ProgrammingError:
            pass


class Embedding(BaseEmbeddingProvider):
    @property
    def dimension(self) -> int:
        return 2

    @property
    def model_name(self) -> str:
        return "test"

    def embed_text(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class Client:
    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self.embeddings = SimpleNamespace(create=self.create_embedding)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create_chat)
        )

    def create_embedding(self, model: str, input: str) -> object:
        return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0])])

    def create_chat(self, **kwargs: object) -> object:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
        )


def test_foundry_model_path_is_loaded_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FOUNDRY_MODEL_PATH", "/models")
    assert Settings.from_env().foundry_model_path == "/models"


def test_streamlit_secrets_bridge_populates_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAG_MODE", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "streamlit",
        SimpleNamespace(
            secrets={
                "RAG_MODE": "AZURE_CLOUD",
                "AZURE_OPENAI_ENDPOINT": "https://ci.example.test/",
            }
        ),
    )
    config._load_streamlit_secrets()
    assert Settings.from_env().rag_mode == "AZURE_CLOUD"
    assert Settings.from_env().azure_endpoint == "https://ci.example.test/"


@pytest.mark.parametrize("query", ["   ", "\x00"])
def test_query_schema_rejects_malformed_values(query: str) -> None:
    with pytest.raises(ValueError):
        QueryRequest(query=query)


def test_refusal_does_not_return_irrelevant_sources() -> None:
    class Chat(BaseChatProvider):
        @property
        def model_name(self) -> str:
            return "chat"

        def generate_response(
            self,
            system_prompt: str,
            user_prompt: str,
            max_tokens: int,
            temperature: float,
        ) -> str:
            return "unused"

    class LowScoreStore(BaseVectorStore):
        def save_document(
            self, source_id: str, chunk_index: int, text: str, embedding: list[float]
        ) -> None:
            pass

        def clear_all(self) -> None:
            pass

        def search_similar(
            self, query_embedding: list[float], top_k: int
        ) -> list[SearchResult]:
            return [
                SearchResult(
                    source_id="unrelated",
                    chunk_index=0,
                    text="private context",
                    score=0.1,
                )
            ]

    response = RAGService(
        Embedding(),
        Chat(),
        LowScoreStore(),
        confidence_threshold=0.9,
    ).query(RAGQueryRequest(query="unknown"))
    assert response.sources == []
    assert response.citations == []


def test_cli_flag_aliases_parse() -> None:
    parser = main.build_parser()
    assert parser.parse_args(["--ingest"]).ingest_flag is True
    assert parser.parse_args(["--query", "question"]).query_flag == "question"
    assert parser.parse_args(["--health"]).health_flag is True


def test_local_providers_pass_model_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "providers.local_foundry.importlib.import_module",
        lambda name: SimpleNamespace(FoundryLocalClient=Client),
    )
    embedding = LocalFoundryEmbeddingProvider(dimension=2, model_path="/models")
    chat = LocalFoundryChatProvider(model_path="/models")
    assert embedding.embed_text("text") == [1.0, 0.0]
    assert chat.generate_response("system", "user", 10, 0.0) == "answer"


def test_ingestion_stores_relative_source_file(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "doc.txt").write_text("content", encoding="utf-8")
    store = SQLiteVectorStore(str(tmp_path / "rag.db"))
    ingest_directory(data, Embedding(), store)
    result = store.search_similar([1.0, 0.0], 1)[0]
    assert result.source_file == "doc.txt"


@pytest.mark.parametrize("raw_embedding", [b"nan,0.0", b"inf,1.0", b"0.0,0.0"])
def test_health_rejects_invalid_persisted_embedding(
    tmp_path: Path, raw_embedding: bytes
) -> None:
    store = SQLiteVectorStore(str(tmp_path / "rag.db"))
    store.save_document("doc", 0, "text", [1.0, 0.0])
    store._connection.execute(
        "UPDATE document_chunks SET embedding = ?", (raw_embedding,)
    )
    store._connection.commit()
    assert store.check_health() is False


def test_evaluation_uses_fixed_precision_and_answer_terms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "cases.json"
    dataset.write_text(
        '[{"question":"question","relevant_sources":["doc"],'
        '"answer_terms":["SQLite"]}]',
        encoding="utf-8",
    )

    class Service:
        def query(self, request: object) -> RAGResponse:
            return RAGResponse(
                answer="SQLite answer",
                sources=[
                    SearchResult(
                        source_id="doc", chunk_index=0, text="context", score=1.0
                    )
                ],
                citations=["doc"],
                latency_seconds=0.01,
                mode="LOCAL",
            )

    monkeypatch.setattr(evaluate, "get_rag_service", lambda settings: Service())
    result = evaluate.evaluate_dataset(dataset, Settings())
    assert result["precision_at_3"] == pytest.approx(1 / 3)
    assert result["faithfulness_citation_grounding"] == 1.0


def test_azure_embedding_batch_and_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response(BytesIO):
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> bool:
            return False

    responses = [
        {"data": [{"embedding": [1.0, 0.0]}]},
        {"data": [{"embedding": [0.0, 1.0]}]},
        {"choices": [{"message": {"content": "answer"}}]},
    ]

    def fake_urlopen(request: object, timeout: int) -> Response:
        return Response(json.dumps(responses.pop(0)).encode())

    monkeypatch.setattr(azure, "urlopen", fake_urlopen)
    embedding = AzureOpenAIEmbeddingProvider("https://example", "key", "v", "e", 2)
    chat = AzureOpenAIChatProvider("https://example", "key", "v", "c")
    assert embedding.embed_batch(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert chat.generate_response("s", "u", 10, 0.0) == "answer"


@pytest.mark.parametrize(
    ("code", "error_type"),
    [
        (401, AzureAuthenticationError),
        (403, AzureAuthorizationError),
        (500, AzureServerError),
    ],
)
def test_azure_http_errors(
    monkeypatch: pytest.MonkeyPatch, code: int, error_type: type[Exception]
) -> None:
    def fake_urlopen(request: object, timeout: int) -> None:
        raise HTTPError("url", code, "error", Message(), BytesIO())

    monkeypatch.setattr(azure, "urlopen", fake_urlopen)
    provider = AzureOpenAIEmbeddingProvider("https://example", "key", "v", "e", 2)
    with pytest.raises(error_type):
        provider.embed_text("text")


def test_azure_rate_limit_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    headers = Message()
    headers["Retry-After"] = "0"

    def fake_urlopen(request: object, timeout: int) -> None:
        raise HTTPError("url", 429, "rate", headers, BytesIO())

    monkeypatch.setattr(azure, "urlopen", fake_urlopen)
    monkeypatch.setattr(azure.time, "sleep", lambda seconds: None)
    provider = AzureOpenAIEmbeddingProvider("https://example", "key", "v", "e", 2)
    with pytest.raises(AzureRateLimitError):
        provider.embed_text("text")


def test_store_metadata_and_health_corruption(tmp_path: Path) -> None:
    store = SQLiteVectorStore(str(tmp_path / "rag.db"))
    store.replace_all_atomic(
        [DocumentChunk(source_id="doc", chunk_index=0, text="text", embedding=[1.0])],
        {
            "embedding_model": "mock",
            "vector_dimension": 1,
            "chunking_config": {},
            "ingestion_timestamp": "now",
        },
    )
    assert store.get_index_metadata() is not None
    store._connection.execute(
        "UPDATE index_metadata SET chunking_config = ?", ("invalid",)
    )
    store._connection.commit()
    assert store.check_health() is False
    store.clear_all()
    assert store.get_index_metadata() is None


def test_local_provider_rejects_empty_and_wrong_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BadClient(Client):
        def create_embedding(self, model: str, input: str) -> object:
            return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0])])

    monkeypatch.setattr(
        "providers.local_foundry.importlib.import_module",
        lambda name: SimpleNamespace(FoundryLocalClient=BadClient),
    )
    provider = LocalFoundryEmbeddingProvider(dimension=2)
    with pytest.raises(ValueError):
        provider.embed_text(" ")
    with pytest.raises(RuntimeError, match="dimensions"):
        provider.embed_text("text")


def test_azure_provider_handles_network_and_malformed_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        azure,
        "urlopen",
        lambda request, timeout: (_ for _ in ()).throw(URLError("offline")),
    )
    provider = AzureOpenAIEmbeddingProvider("https://example", "key", "v", "e", 2)
    with pytest.raises(azure.AzureProviderError):
        provider.embed_text("text")

    class Response(BytesIO):
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> bool:
            return False

    monkeypatch.setattr(azure, "urlopen", lambda request, timeout: Response(b"{}"))
    with pytest.raises(RuntimeError, match="unsupported shape"):
        provider.embed_text("text")


def test_store_rejects_invalid_write_and_search_arguments(tmp_path: Path) -> None:
    store = SQLiteVectorStore(str(tmp_path / "rag.db"), vector_dimension=2)
    with pytest.raises(ValueError):
        store.save_document("doc", 0, "text", [1.0])
    with pytest.raises(ValueError):
        store.save_document("", 0, "text", [1.0, 0.0])
    with pytest.raises(ValueError):
        store.search_similar([1.0, 0.0], 0)


def test_local_client_import_fallback_and_missing_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def import_module(name: str) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ImportError("sdk name differs")
        return SimpleNamespace(Client=Client)

    monkeypatch.setattr(
        "providers.local_foundry.importlib.import_module", import_module
    )
    provider = LocalFoundryEmbeddingProvider(dimension=2)
    assert provider.embed_text("text") == [1.0, 0.0]

    monkeypatch.setattr(
        "providers.local_foundry.importlib.import_module",
        lambda name: SimpleNamespace(),
    )
    with pytest.raises(RuntimeError, match="does not expose"):
        provider.embed_text("text")


def test_local_client_initialization_failure_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "providers.local_foundry.importlib.import_module",
        lambda name: (_ for _ in ()).throw(ImportError("missing sdk")),
    )
    provider = LocalFoundryEmbeddingProvider(dimension=2, model_path="/missing")
    with pytest.raises(RuntimeError, match="Foundry embedding failed"):
        provider.embed_text("text")


def test_local_invalid_embedding_response_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidClient(Client):
        def create_embedding(self, model: str, input: str) -> object:
            return {"unexpected": "shape"}

    monkeypatch.setattr(
        "providers.local_foundry.importlib.import_module",
        lambda name: SimpleNamespace(Client=InvalidClient),
    )
    provider = LocalFoundryEmbeddingProvider(dimension=2)
    with pytest.raises(RuntimeError, match="unsupported shape"):
        provider.embed_text("text")


def test_sqlite_corrupt_blob_fails_search(tmp_path: Path) -> None:
    store = SQLiteVectorStore(str(tmp_path / "rag.db"))
    store.save_document("doc", 0, "text", [1.0, 0.0])
    store._connection.execute(
        "UPDATE document_chunks SET embedding = ?", (b"not-a-vector",)
    )
    store._connection.commit()
    with pytest.raises(ValueError):
        store.search_similar([1.0, 0.0], 1)


def test_sqlite_closed_connection_errors_are_structured(tmp_path: Path) -> None:
    store = SQLiteVectorStore(str(tmp_path / "rag.db"))
    store.close()
    assert store.check_health() is False
    assert store.get_index_metadata() is None
    with pytest.raises(RuntimeError):
        store.search_similar([1.0], 1)


def test_evaluation_empty_dataset_and_zero_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "empty.json"
    dataset.write_text("[]", encoding="utf-8")

    class EmptyService:
        def query(self, request: object) -> RAGResponse:
            return RAGResponse(
                answer="no result",
                sources=[],
                citations=[],
                latency_seconds=0.0,
                mode="LOCAL",
            )

    monkeypatch.setattr(evaluate, "get_rag_service", lambda settings: EmptyService())
    empty = evaluate.evaluate_dataset(dataset, Settings())
    assert empty["case_count"] == 0

    dataset.write_text(
        '[{"question":"question","relevant_sources":[]}]', encoding="utf-8"
    )
    result = evaluate.evaluate_dataset(dataset, Settings())
    assert result["precision_at_3"] == 0.0
    assert result["mrr"] == 0.0
    assert result["faithfulness_citation_grounding"] == 0.0


def test_store_metadata_fallback_and_atomic_metadata_validation(tmp_path: Path) -> None:
    store = SQLiteVectorStore(str(tmp_path / "rag.db"))
    store.save_document("doc", 0, "text", [1.0, 0.0])
    assert store.search_similar([1.0, 0.0], 1)[0].source_id == "doc"
    with pytest.raises(ValueError):
        store.replace_all_atomic([], {})
    with pytest.raises(ValueError):
        store.replace_all_atomic(
            [
                DocumentChunk(
                    source_id="doc", chunk_index=0, text="text", embedding=[1.0, 0.0]
                )
            ],
            {
                "embedding_model": "mock",
                "vector_dimension": 0,
                "chunking_config": {},
                "ingestion_timestamp": "now",
            },
        )
    with pytest.raises(ValueError):
        store.replace_all_atomic(
            [
                DocumentChunk(
                    source_id="doc", chunk_index=0, text="text", embedding=[1.0, 0.0]
                )
            ],
            {"embedding_model": "mock", "vector_dimension": 2, "chunking_config": {}},
        )


def test_cosine_rejects_dimension_mismatch_and_handles_zero_candidate() -> None:
    assert SQLiteVectorStore._cosine([1.0], 1.0, [0.0]) == 0.0
    with pytest.raises(ValueError, match="dimensions"):
        SQLiteVectorStore._cosine([1.0], 1.0, [1.0, 0.0])


def test_sqlite_initialization_and_save_failures_are_structured(tmp_path: Path) -> None:
    nested_database = tmp_path / "missing" / "rag.db"
    nested_store = SQLiteVectorStore(str(nested_database))
    assert nested_database.exists()
    nested_store.close()

    store = SQLiteVectorStore(str(tmp_path / "rag.db"))
    store.save_document("first", 0, "text", [1.0, 0.0])
    store.save_document("second", 0, "text", [0.0, 1.0])
    assert store.get_index_metadata()["total_chunk_count"] == 2  # type: ignore[index]
    store.close()
    with pytest.raises(RuntimeError, match="save document"):
        store.save_document("third", 0, "text", [1.0, 0.0])
