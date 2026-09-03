"""SQLite-backed vector store with BLOB embeddings and normalized cosine similarity."""

import json
import math
import sqlite3
from collections.abc import Iterator
from threading import RLock

from core.models import DocumentChunk, SearchResult
from core.ports import BaseVectorStore


class SQLiteVectorStore(BaseVectorStore):
    """Persist embeddings as ASCII-encoded BLOBs and calculate cosine scores in Python."""

    def __init__(
        self,
        database_path: str = "rag.db",
        vector_dimension: int | None = None,
        embedding_model: str = "unknown",
    ) -> None:
        self.database_path = database_path
        self.vector_dimension = vector_dimension
        self.embedding_model = embedding_model
        self._lock = RLock()
        try:
            self._connection = sqlite3.connect(
                database_path, timeout=30.0, check_same_thread=False
            )
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.execute("PRAGMA busy_timeout=30000")
            self._connection.execute(
                """CREATE TABLE IF NOT EXISTS document_chunks (
                    source_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    PRIMARY KEY (source_id, chunk_index)
                )"""
            )
            self._connection.execute("""CREATE TABLE IF NOT EXISTS index_metadata (
                id INTEGER PRIMARY KEY CHECK (id = 1), embedding_model TEXT NOT NULL,
                vector_dimension INTEGER NOT NULL, chunking_config TEXT NOT NULL,
                total_chunk_count INTEGER NOT NULL, ingestion_timestamp TEXT NOT NULL)""")
            self._connection.execute("""CREATE TABLE IF NOT EXISTS chunk_metadata (
                source_id TEXT NOT NULL, chunk_index INTEGER NOT NULL,
                source_file TEXT, character_offset INTEGER NOT NULL, corpus_hash TEXT,
                PRIMARY KEY (source_id, chunk_index))""")
            self._connection.commit()
        except sqlite3.Error as exc:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            raise RuntimeError(f"Could not initialize SQLite store: {exc}") from exc

    def save_document(
        self, source_id: str, chunk_index: int, text: str, embedding: list[float]
    ) -> None:
        """Insert or replace a chunk, rejecting empty or invalid vectors."""
        self._validate_vector(embedding)
        if (
            self.vector_dimension is not None
            and len(embedding) != self.vector_dimension
        ):
            raise ValueError(
                "embedding dimension does not match the configured provider"
            )
        if not source_id or not text:
            raise ValueError("source_id and text must not be empty")
        if chunk_index < 0:
            raise ValueError("chunk_index must not be negative")
        try:
            with self._lock:
                self._connection.execute("BEGIN")
                exists = (
                    self._connection.execute(
                        "SELECT 1 FROM document_chunks WHERE source_id = ? AND chunk_index = ?",
                        (source_id, chunk_index),
                    ).fetchone()
                    is not None
                )
                self._connection.execute(
                    "INSERT OR REPLACE INTO document_chunks VALUES (?, ?, ?, ?)",
                    (
                        source_id,
                        chunk_index,
                        text,
                        self._encode(embedding),
                    ),
                )
                self._connection.execute(
                    "DELETE FROM chunk_metadata WHERE source_id = ? AND chunk_index = ?",
                    (source_id, chunk_index),
                )
                self._connection.execute(
                    "INSERT INTO chunk_metadata VALUES (?, ?, ?, ?, ?)",
                    (source_id, chunk_index, None, 0, None),
                )
                metadata = self._connection.execute(
                    "SELECT vector_dimension, total_chunk_count FROM index_metadata WHERE id = 1"
                ).fetchone()
                if metadata is None:
                    self._connection.execute(
                        "INSERT INTO index_metadata VALUES (1, ?, ?, ?, ?, ?)",
                        (self.embedding_model, len(embedding), "{}", 1, "manual"),
                    )
                elif metadata[0] != len(embedding):
                    raise ValueError(
                        "embedding dimension does not match index metadata"
                    )
                elif not exists:
                    self._connection.execute(
                        "UPDATE index_metadata SET total_chunk_count = total_chunk_count + 1 WHERE id = 1"
                    )
                self._connection.commit()
        except BaseException as exc:
            try:
                with self._lock:
                    self._connection.rollback()
            except BaseException:
                pass
            if isinstance(exc, ValueError):
                raise exc
            if isinstance(exc, Exception):
                raise RuntimeError(f"Could not save document chunk: {exc}") from exc
            raise

    def search_similar(
        self, query_embedding: list[float], top_k: int
    ) -> list[SearchResult]:
        """Return normalized cosine similarity results in descending order."""
        self._validate_vector(query_embedding)
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        expected_dimension = self.vector_dimension
        if expected_dimension is None:
            try:
                with self._lock:
                    row = self._connection.execute(
                        "SELECT vector_dimension FROM index_metadata WHERE id = 1"
                    ).fetchone()
                expected_dimension = row[0] if row is not None else None
            except sqlite3.Error as exc:
                raise RuntimeError(f"Could not read index metadata: {exc}") from exc
        if (
            expected_dimension is not None
            and len(query_embedding) != expected_dimension
        ):
            raise ValueError("query embedding dimension does not match the index")
        query_norm = math.hypot(*query_embedding)
        if query_norm == 0:
            raise ValueError("query_embedding must not be a zero vector")
        try:
            with self._lock:
                rows: Iterator[
                    tuple[int, str, int, str, bytes, str | None, int | None, str | None]
                ] = iter(
                    self._connection.execute(
                        "SELECT dc.rowid, dc.source_id, dc.chunk_index, dc.text, dc.embedding, "
                        "cm.source_file, cm.character_offset, cm.corpus_hash "
                        "FROM document_chunks AS dc LEFT JOIN chunk_metadata AS cm "
                        "ON cm.source_id = dc.source_id AND cm.chunk_index = dc.chunk_index"
                    )
                )
                results = [
                    (
                        rowid,
                        SearchResult(
                            source_id=source_id,
                            chunk_index=chunk_index,
                            text=text,
                            score=self._cosine(
                                query_embedding, query_norm, self._decode(embedding)
                            ),
                            source_file=source_file,
                            character_offset=character_offset,
                            corpus_hash=corpus_hash,
                        ),
                    )
                    for rowid, source_id, chunk_index, text, embedding, source_file, character_offset, corpus_hash in rows
                ]
        except sqlite3.Error as exc:
            raise RuntimeError(f"Could not search document chunks: {exc}") from exc
        return [
            result
            for _, result in sorted(
                results,
                key=lambda item: (
                    -item[1].score,
                    item[1].source_id,
                    item[1].chunk_index,
                    item[0],
                ),
            )[:top_k]
        ]

    def clear_all(self) -> None:
        """Delete all chunks from the store."""
        try:
            with self._lock:
                self._connection.execute("BEGIN")
                self._connection.execute("DELETE FROM document_chunks")
                self._connection.execute("DELETE FROM chunk_metadata")
                self._connection.execute("DELETE FROM index_metadata")
                self._connection.commit()
        except BaseException as exc:
            try:
                with self._lock:
                    self._connection.rollback()
            except BaseException:
                pass
            if isinstance(exc, Exception):
                raise RuntimeError(f"Could not clear SQLite store: {exc}") from exc
            raise

    def replace_all_atomic(
        self, chunks: list[DocumentChunk], metadata: dict[str, object]
    ) -> None:
        """Replace chunks and metadata in one transaction, rolling back on failure."""
        if not chunks:
            raise ValueError("cannot replace the index with no chunks")
        required = {
            "embedding_model",
            "vector_dimension",
            "chunking_config",
            "ingestion_timestamp",
        }
        if not required.issubset(metadata):
            raise ValueError(f"metadata must contain {sorted(required)}")
        dimension = metadata["vector_dimension"]
        if not isinstance(dimension, int) or dimension < 1:
            raise ValueError("vector_dimension must be a positive integer")
        if self.vector_dimension is not None and dimension != self.vector_dimension:
            raise ValueError("vector_dimension does not match the configured provider")
        try:
            with self._lock:
                self._connection.execute("BEGIN")
                self._connection.execute("DELETE FROM document_chunks")
                self._connection.execute("DELETE FROM chunk_metadata")
                for chunk in chunks:
                    if chunk.embedding is None:
                        raise ValueError("every chunk must have an embedding")
                    self._validate_vector(chunk.embedding)
                    if len(chunk.embedding) != dimension:
                        raise ValueError(
                            "chunk embedding dimension does not match metadata"
                        )
                    self._connection.execute(
                        "INSERT INTO document_chunks VALUES (?, ?, ?, ?)",
                        (
                            chunk.source_id,
                            chunk.chunk_index,
                            chunk.text,
                            self._encode(chunk.embedding),
                        ),
                    )
                    self._connection.execute(
                        "INSERT INTO chunk_metadata VALUES (?, ?, ?, ?, ?)",
                        (
                            chunk.source_id,
                            chunk.chunk_index,
                            chunk.source_file,
                            chunk.character_offset,
                            chunk.corpus_hash,
                        ),
                    )
                config = json.dumps(metadata["chunking_config"], sort_keys=True)
                self._connection.execute("DELETE FROM index_metadata")
                self._connection.execute(
                    "INSERT INTO index_metadata VALUES (1, ?, ?, ?, ?, ?)",
                    (
                        metadata["embedding_model"],
                        metadata["vector_dimension"],
                        config,
                        len(chunks),
                        metadata["ingestion_timestamp"],
                    ),
                )
                self._connection.commit()
                self.vector_dimension = dimension
        except BaseException as exc:
            try:
                with self._lock:
                    self._connection.rollback()
            except BaseException:
                pass
            if isinstance(exc, ValueError):
                raise exc
            if isinstance(exc, Exception):
                raise RuntimeError(
                    f"Could not atomically replace index: {exc}"
                ) from exc
            raise

    def get_index_metadata(self) -> dict[str, object] | None:
        """Return latest ingestion metadata, if available."""
        try:
            with self._lock:
                row = self._connection.execute(
                    "SELECT embedding_model, vector_dimension, chunking_config, total_chunk_count, ingestion_timestamp FROM index_metadata WHERE id = 1"
                ).fetchone()
        except sqlite3.Error:
            return None
        if row is None:
            return None
        try:
            chunking_config = json.loads(row[2])
        except (json.JSONDecodeError, TypeError):
            return None
        return {
            "embedding_model": row[0],
            "vector_dimension": row[1],
            "chunking_config": chunking_config,
            "total_chunk_count": row[3],
            "ingestion_timestamp": row[4],
        }

    def check_health(self) -> bool:
        """Verify that the vector store tables can be queried."""
        try:
            with self._lock:
                tables = {
                    row[0]
                    for row in self._connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
                if not {"document_chunks", "chunk_metadata", "index_metadata"}.issubset(
                    tables
                ):
                    return False
                chunk_count = self._connection.execute(
                    "SELECT COUNT(*) FROM document_chunks"
                ).fetchone()[0]
                row = self._connection.execute(
                    "SELECT vector_dimension, chunking_config, total_chunk_count "
                    "FROM index_metadata WHERE id = 1"
                ).fetchone()
                if chunk_count == 0 or row is None:
                    return False
                if chunk_count != row[2]:
                    return False
                embedding_row = self._connection.execute(
                    "SELECT embedding FROM document_chunks LIMIT 1"
                ).fetchone()
                if embedding_row is None:
                    return False
                try:
                    embedding = self._decode(embedding_row[0])
                except (AttributeError, TypeError, UnicodeDecodeError, ValueError):
                    return False
                try:
                    self._validate_vector(embedding)
                except ValueError:
                    return False
                expected_dimension = (
                    self.vector_dimension
                    if self.vector_dimension is not None
                    else row[0]
                )
                if (
                    self.vector_dimension is not None
                    and row[0] != self.vector_dimension
                ):
                    raise ValueError(
                        "stored vector dimension does not match the configured provider"
                    )
                if len(embedding) != expected_dimension:
                    raise ValueError(
                        "stored embedding dimension does not match the index"
                    )
                chunking_config = json.loads(row[1])
                if not isinstance(chunking_config, dict):
                    return False
            return True
        except (sqlite3.Error, json.JSONDecodeError, TypeError):
            return False

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            try:
                self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass
            self._connection.close()

    def __enter__(self) -> "SQLiteVectorStore":
        """Return the store for a managed database lifetime."""
        return self

    def __exit__(
        self, _exc_type: object, _exc_value: object, _traceback: object
    ) -> None:
        """Close the database connection on context exit."""
        self.close()

    @staticmethod
    def _validate_vector(vector: list[float]) -> None:
        if (
            not vector
            or any(
                not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in vector
            )
            or not any(value != 0 for value in vector)
        ):
            raise ValueError(
                "embedding must contain finite values and must not be a zero vector"
            )

    @staticmethod
    def _encode(vector: list[float]) -> bytes:
        return ",".join(str(value) for value in vector).encode("ascii")

    @staticmethod
    def _decode(data: bytes) -> list[float]:
        return [float(value) for value in data.decode("ascii").split(",")]

    @staticmethod
    def _cosine(query: list[float], query_norm: float, candidate: list[float]) -> float:
        if len(query) != len(candidate):
            raise ValueError("query and candidate dimensions must match")
        candidate_norm = math.hypot(*candidate)
        if candidate_norm == 0:
            return 0.0
        score = sum(
            (left / query_norm) * (right / candidate_norm)
            for left, right in zip(query, candidate)
        )
        return max(-1.0, min(1.0, score))
