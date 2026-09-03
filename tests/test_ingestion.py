import sqlite3
from pathlib import Path
from typing import Any, cast

import pytest

from core.ports import BaseEmbeddingProvider
from ingestion import ingest_directory
from stores.sqlite_store import SQLiteVectorStore


@pytest.fixture(autouse=True)
def close_sqlite_stores(monkeypatch: pytest.MonkeyPatch):
    """Close stores created by ingestion tests after each test."""
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


class FailingEmbedding(BaseEmbeddingProvider):
    @property
    def dimension(self) -> int:
        return 2

    @property
    def model_name(self) -> str:
        return "failing"

    def embed_text(self, _text: str) -> list[float]:
        raise RuntimeError("boom")

    def embed_batch(self, _texts: list[str]) -> list[list[float]]:
        raise RuntimeError("boom")


class GoodEmbedding(FailingEmbedding):
    def embed_text(self, _text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def test_binary_files_are_skipped(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "binary.txt").write_bytes(b"\x00\xff")
    with pytest.raises(ValueError, match="no readable content"):
        ingest_directory(data, GoodEmbedding(), SQLiteVectorStore(str(tmp_path / "db")))


def test_provider_failure_preserves_existing_index(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "new.txt").write_text("new content", encoding="utf-8")
    store = SQLiteVectorStore(str(tmp_path / "db"))
    store.save_document("old", 0, "old content", [1.0, 0.0])
    with pytest.raises(RuntimeError, match="existing index was preserved"):
        ingest_directory(data, FailingEmbedding(), store)
    assert store.search_similar([1.0, 0.0], 1)[0].source_id == "old"


def test_embedding_dimension_mismatch_preserves_existing_index(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "new.txt").write_text("new content", encoding="utf-8")
    store = SQLiteVectorStore(str(tmp_path / "db"))
    store.save_document("old", 0, "old content", [1.0, 0.0])
    provider = GoodEmbedding()
    provider.embed_batch = lambda texts: [[1.0] for _ in texts]
    with pytest.raises(RuntimeError, match="vector dimension"):
        ingest_directory(data, provider, store)
    assert store.search_similar([1.0, 0.0], 1)[0].source_id == "old"
