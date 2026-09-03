"""Atomic ingestion orchestration for the hybrid RAG index."""

import logging
import os
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from core.chunking import DocumentChunker
from core.models import DocumentChunk
from core.ports import BaseEmbeddingProvider, BaseVectorStore

SUPPORTED_TEXT_SUFFIXES = {".txt", ".md"}
logger = logging.getLogger(__name__)


def _snapshot_directory(directory: Path) -> tuple[str, list[tuple[Path, str]]]:
    """Read valid files once and return their deterministic hash and contents."""
    records: list[tuple[str, bytes, str, Path]] = []
    for current_root, directory_names, file_names in os.walk(directory):
        current_path = Path(current_root)
        directory_names[:] = [
            name
            for name in directory_names
            if not name.startswith(".")
            and not (current_path / name).is_symlink()
        ]
        for filename in sorted(file_names):
            path = current_path / filename
            relative_name = path.relative_to(directory).as_posix()
            if (
                filename.startswith(".")
                or path.is_symlink()
                or not path.is_file()
                or path.suffix.lower() not in SUPPORTED_TEXT_SUFFIXES
            ):
                continue
            try:
                with path.open("rb") as stream:
                    raw_content = stream.read()
                if b"\x00" in raw_content:
                    logger.warning(
                        "Skipping binary document",
                        extra={"path": str(path), "reason": "null_byte"},
                    )
                    continue
                content = raw_content.decode("utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                logger.warning(
                    "Skipping unreadable document",
                    extra={"path": str(path), "reason": type(exc).__name__},
                )
                continue
            records.append((relative_name, raw_content, content, path))

    digest = sha256()
    contents: list[tuple[Path, str]] = []
    for filename_text, raw_content, content, path in sorted(records):
        filename_bytes = filename_text.encode("utf-8")
        digest.update(f"{len(filename_bytes)}:".encode("ascii"))
        digest.update(filename_bytes)
        digest.update(f":{len(raw_content)}:".encode("ascii"))
        digest.update(raw_content)
        digest.update(b":")
        contents.append((path, content))
    return digest.hexdigest(), contents


def calculate_corpus_hash(data_dir: str | Path) -> str:
    """Return the deterministic hash used for a directory ingestion."""
    directory = Path(data_dir)
    if not directory.is_dir():
        raise ValueError(f"data directory does not exist: {directory}")
    corpus_hash, _ = _snapshot_directory(directory)
    return corpus_hash


def ingest_directory(
    data_dir: str | Path,
    embedding_provider: BaseEmbeddingProvider,
    vector_store: BaseVectorStore,
    chunker: DocumentChunker | None = None,
    corpus_hash_out: list[str] | None = None,
) -> int:
    """Build a complete index from ``data_dir`` without touching the old index on failure."""
    directory = Path(data_dir)
    if not directory.is_dir():
        raise ValueError(f"data directory does not exist: {directory}")
    corpus_hash, valid_contents = _snapshot_directory(directory)
    if corpus_hash_out is not None:
        corpus_hash_out.append(corpus_hash)
    resolved_chunker = chunker or DocumentChunker()
    chunks: list[DocumentChunk] = []
    for path, content in valid_contents:
        source_name = path.relative_to(directory).as_posix()
        chunks.extend(
            resolved_chunker.chunk_text(
                content,
                source_name,
                source_file=source_name,
                corpus_hash=corpus_hash,
            )
        )
    if not chunks:
        raise ValueError(
            "data directory contains no readable content; existing index was preserved"
        )
    texts = [chunk.text for chunk in chunks]
    try:
        embeddings = embedding_provider.embed_batch(texts)
    except Exception as exc:
        raise RuntimeError(
            f"embedding generation failed; existing index was preserved: {exc}"
        ) from exc
    if len(embeddings) != len(chunks):
        raise RuntimeError("embedding provider returned an unexpected batch length")
    if any(len(embedding) != embedding_provider.dimension for embedding in embeddings):
        raise RuntimeError("embedding provider returned an unexpected vector dimension")
    embedded = [
        chunk.model_copy(update={"embedding": embedding})
        for chunk, embedding in zip(chunks, embeddings)
    ]
    metadata: dict[str, object] = {
        "embedding_model": embedding_provider.model_name,
        "vector_dimension": embedding_provider.dimension,
        "chunking_config": resolved_chunker.config_dict(),
        "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    vector_store.replace_all_atomic(embedded, metadata)
    return len(embedded)
