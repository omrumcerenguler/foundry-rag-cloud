import sqlite3
from pathlib import Path
from typing import Any, cast

import pytest

from core.models import DocumentChunk
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


def test_cosine_similarity_ranking(tmp_path: Path) -> None:
    store = SQLiteVectorStore(str(tmp_path / "rag.db"))
    store.save_document("near", 0, "near", [1.0, 0.0])
    store.save_document("far", 0, "far", [0.0, 1.0])
    results = store.search_similar([1.0, 0.0], 2)
    assert [result.source_id for result in results] == ["near", "far"]


def test_database_parent_directories_are_created(tmp_path: Path) -> None:
    database = tmp_path / "nested" / "stores" / "rag.db"
    store = SQLiteVectorStore(str(database))
    assert database.exists()
    store.close()


def test_atomic_replacement_rolls_back(tmp_path: Path) -> None:
    store = SQLiteVectorStore(str(tmp_path / "rag.db"))
    store.save_document("old", 0, "old", [1.0])
    chunks = [
        DocumentChunk(source_id="new", chunk_index=0, text="new", embedding=[1.0]),
        DocumentChunk(source_id="broken", chunk_index=1, text="broken"),
    ]
    with pytest.raises(ValueError):
        store.replace_all_atomic(
            chunks,
            {
                "embedding_model": "mock",
                "vector_dimension": 1,
                "chunking_config": {},
                "ingestion_timestamp": "now",
            },
        )
    assert store.search_similar([1.0], 5)[0].source_id == "old"


def test_zero_norm_and_dimension_mismatch_are_safe(tmp_path: Path) -> None:
    store = SQLiteVectorStore(str(tmp_path / "rag.db"))
    with pytest.raises(ValueError, match="zero vector"):
        store.save_document("zero", 0, "zero", [0.0, 0.0])
    with pytest.raises(ValueError, match="zero vector"):
        store.search_similar([0.0, 0.0], 1)
    store.save_document("valid", 0, "valid", [1.0, 0.0])
    with pytest.raises(ValueError, match="dimension"):
        store.search_similar([1.0], 1)
    with pytest.raises(ValueError, match="dimension"):
        store.replace_all_atomic(
            [
                DocumentChunk(
                    source_id="bad", chunk_index=0, text="bad", embedding=[1.0]
                )
            ],
            {
                "embedding_model": "mock",
                "vector_dimension": 2,
                "chunking_config": {},
                "ingestion_timestamp": "now",
            },
        )


def test_wal_busy_timeout_context_and_deterministic_ties(tmp_path: Path) -> None:
    database = tmp_path / "rag.db"
    with SQLiteVectorStore(str(database)) as store:
        assert store._connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert store._connection.execute("PRAGMA busy_timeout").fetchone()[0] == 30000
        store.save_document("first", 0, "first", [1.0, 0.0])
        store.save_document("second", 0, "second", [1.0, 0.0])
        assert [result.source_id for result in store.search_similar([1.0, 0.0], 2)] == [
            "first",
            "second",
        ]
    with pytest.raises(RuntimeError):
        store.search_similar([1.0, 0.0], 1)


def test_clear_all_removes_index_metadata(tmp_path: Path) -> None:
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
    store.clear_all()
    assert store.get_index_metadata() is None


def test_health_rejects_chunk_count_mismatch(tmp_path: Path) -> None:
    store = SQLiteVectorStore(str(tmp_path / "rag.db"))
    store.save_document("doc", 0, "text", [1.0])
    store._connection.execute(
        "UPDATE index_metadata SET total_chunk_count = 2 WHERE id = 1"
    )
    store._connection.commit()

    assert store.check_health() is False


def test_search_handles_unmapped_chunk_provenance(tmp_path: Path) -> None:
    store = SQLiteVectorStore(str(tmp_path / "rag.db"))
    store.save_document("doc", 0, "text", [1.0])
    store._connection.execute(
        "DELETE FROM chunk_metadata WHERE source_id = ? AND chunk_index = ?",
        ("doc", 0),
    )
    store._connection.commit()

    result = store.search_similar([1.0], 1)[0]
    assert result.source_file is None
    assert result.character_offset is None
    assert result.corpus_hash is None
