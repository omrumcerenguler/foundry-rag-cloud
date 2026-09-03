"""HTTP contract tests for the production API."""

from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import api
from core.models import RAGResponse, SearchResult
from providers.azure_openai import AzureRateLimitError


def _service(
    answer: str = "Grounded [Source: doc1]", total_chunks: int = 1
) -> SimpleNamespace:
    """Build a small service-shaped test double."""
    result = SearchResult(
        source_id="doc1",
        chunk_index=0,
        text="context",
        score=0.9,
        source_file="data/doc1.txt",
        character_offset=12,
        corpus_hash="corpus-123",
    )
    return SimpleNamespace(
        mode="LOCAL",
        embedding_provider=SimpleNamespace(model_name="mock", dimension=2),
        vector_store=SimpleNamespace(
            get_index_metadata=lambda: {
                "embedding_model": "mock",
                "vector_dimension": 2,
                "total_chunk_count": total_chunks,
            },
            close=lambda: None,
        ),
        query=lambda request: RAGResponse(
            answer=answer,
            citations=["doc1"],
            sources=[result],
            latency_seconds=0.01,
            mode="LOCAL",
        ),
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    """Use a deterministic service while preserving the real lifespan."""
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setattr(api, "get_rag_service", lambda settings: _service())
    with TestClient(api.app) as test_client:
        yield test_client


def test_health_returns_active_index_info(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["total_chunks"] == 1


def test_query_returns_grounded_sources(client: TestClient) -> None:
    response = client.post("/query", json={"query": "known"})
    assert response.status_code == 200
    assert response.json()["citations"] == ["doc1"]
    assert response.json()["confidence_score"] == 0.9
    assert response.json()["sources"][0]["source_file"] == "data/doc1.txt"
    assert response.json()["sources"][0]["character_offset"] == 12
    assert response.json()["sources"][0]["corpus_hash"] == "corpus-123"


def test_validation_error_is_structured(client: TestClient) -> None:
    response = client.post("/query", json={"query": "", "top_k": 0})
    assert response.status_code == 422
    assert "error" in response.json()


@pytest.mark.parametrize("query", ["   ", "\x00"])
def test_malformed_query_is_rejected_with_422(client: TestClient, query: str) -> None:
    response = client.post("/query", json={"query": query})
    assert response.status_code == 422


def test_rate_limit_is_mapped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_KEY", raising=False)
    service = _service()
    service.query = lambda request: (_ for _ in ()).throw(
        AzureRateLimitError("limited")
    )
    monkeypatch.setattr(api, "get_rag_service", lambda settings: service)
    with TestClient(api.app) as client:
        response = client.post("/query", json={"query": "known"})
    assert response.status_code == 429


def test_ingest_returns_atomic_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("API_KEY", raising=False)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "doc.txt").write_text("content", encoding="utf-8")
    monkeypatch.setenv("RAG_DATA_DIR", str(data_dir))
    monkeypatch.setattr(api, "get_rag_service", lambda settings: _service())

    def mock_ingest(*args: object, **kwargs: object) -> int:
        corpus_hash_out = kwargs.get("corpus_hash_out")
        if corpus_hash_out is not None:
            assert isinstance(corpus_hash_out, list)
            corpus_hash_out.append("mock_hash_abc123")
        return 4

    monkeypatch.setattr(api, "ingest_directory", mock_ingest)
    with TestClient(api.app) as client:
        response = client.post("/ingest")
    assert response.status_code == 200
    assert response.json()["chunks_indexed"] == 4
    assert response.json()["corpus_hash"]


def test_api_key_protects_query_ingest_and_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Require the configured API key on every non-health endpoint."""
    monkeypatch.setenv("API_KEY", "test-key")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "doc.txt").write_text("content", encoding="utf-8")
    monkeypatch.setenv("RAG_DATA_DIR", str(data_dir))
    monkeypatch.setattr(api, "get_rag_service", lambda settings: _service())
    monkeypatch.setattr(api, "ingest_directory", lambda *args, **kwargs: 1)
    with TestClient(api.app) as client:
        for method, path in (
            ("post", "/query"),
            ("post", "/ingest"),
            ("get", "/metadata"),
        ):
            request = getattr(client, method)
            payload = (
                {"query": "known"} if method == "post" and path == "/query" else None
            )
            response = request(path, json=payload) if payload else request(path)
            assert response.status_code == 401
            response = (
                request(path, headers={"X-API-Key": "wrong"}, json=payload)
                if payload
                else request(path, headers={"X-API-Key": "wrong"})
            )
            assert response.status_code == 401
            response = (
                request(path, headers={"X-API-Key": "test-key"}, json=payload)
                if payload
                else request(path, headers={"X-API-Key": "test-key"})
            )
            assert response.status_code == 200


@pytest.mark.parametrize("configured_key", ["", "   "])
def test_empty_api_key_disables_authentication(
    monkeypatch: pytest.MonkeyPatch, configured_key: str
) -> None:
    monkeypatch.setenv("API_KEY", configured_key)
    monkeypatch.setattr(api, "get_rag_service", lambda settings: _service())

    with TestClient(api.app) as client:
        response = client.post("/query", json={"query": "known"})

    assert response.status_code == 200


def test_empty_store_health_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report an empty index as HTTP 503 degraded readiness."""
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setattr(
        api, "get_rag_service", lambda settings: _service(total_chunks=0)
    )
    with TestClient(api.app) as client:
        response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


def test_invalid_api_key_precedes_query_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject an invalid key before validating a malformed query body."""
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setattr(api, "get_rag_service", lambda settings: _service())
    with TestClient(api.app) as client:
        response = client.post(
            "/query",
            content=b"{",
            headers={"Content-Type": "application/json", "X-API-Key": "wrong"},
        )
    assert response.status_code == 401


def test_dimension_mismatch_health_is_degraded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report persisted/provider dimension mismatch as HTTP 503."""
    monkeypatch.delenv("API_KEY", raising=False)
    service = _service()
    service.vector_store.check_health = lambda: False
    monkeypatch.setattr(api, "get_rag_service", lambda settings: service)
    with TestClient(api.app) as client:
        response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
