"""Deterministic, metadata-aware document chunking."""

import re
from dataclasses import dataclass
from pathlib import Path

from .models import DocumentChunk


@dataclass(frozen=True)
class ChunkingConfig:
    """Chunking parameters recorded with an index."""

    chunk_size: int = 1000
    character_overlap: int = 100
    token_overlap: int = 0

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.character_overlap < self.chunk_size:
            raise ValueError("character_overlap must be in [0, chunk_size)")
        if not 0 <= self.token_overlap < self.chunk_size:
            raise ValueError("token_overlap must be in [0, chunk_size)")


class DocumentChunker:
    """Split text into bounded, overlapping chunks while retaining provenance."""

    def __init__(
        self,
        chunk_size: int = 1000,
        character_overlap: int = 100,
        token_overlap: int = 0,
        overlap: int | None = None,
    ) -> None:
        self.config = ChunkingConfig(
            chunk_size, character_overlap if overlap is None else overlap, token_overlap
        )

    def chunk_text(
        self,
        text: str,
        source_id: str,
        *,
        source_file: str | None = None,
        corpus_hash: str | None = None,
    ) -> list[DocumentChunk]:
        """Return chunks with exact character offsets and source metadata."""
        if not isinstance(text, str):
            raise TypeError("document text must be a string")
        if not text.strip():
            return []
        chunks: list[DocumentChunk] = []
        seen_pieces: set[str] = set()
        start = index = 0
        while start < len(text):
            previous_start = start
            end = min(len(text), start + self.config.chunk_size)
            if end < len(text):
                whitespace = list(re.finditer(r"\s", text[start + 1:end]))
                if whitespace:
                    boundary = start + 1 + whitespace[-1].start()
                    if boundary > start:
                        end = boundary
            piece = text[start:end]
            if piece.strip() and piece not in seen_pieces:
                seen_pieces.add(piece)
                chunks.append(
                    DocumentChunk(
                        source_id=source_id,
                        chunk_index=index,
                        text=piece,
                        source_file=source_file,
                        character_offset=start,
                        corpus_hash=corpus_hash,
                    )
                )
                index += 1
            if end == len(text):
                break
            start = end - self.config.character_overlap
            if self.config.token_overlap:
                token_starts = [match.start() for match in re.finditer(r"\S+", piece)]
                if token_starts:
                    token_index = max(0, len(token_starts) - self.config.token_overlap)
                    token_start = previous_start + token_starts[token_index]
                    start = min(start, token_start)
            start = max(previous_start + 1, start)
        return chunks

    def chunk_file(
        self,
        path: str | Path,
        content: str | bytes | None = None,
        *,
        corpus_hash: str | None = None,
    ) -> list[DocumentChunk]:
        """Chunk supplied file content, reading the file once when content is absent."""
        file_path = Path(path)
        try:
            raw_content = (
                content
                if isinstance(content, bytes)
                else content.encode("utf-8")
                if isinstance(content, str)
                else None
            )
            if raw_content is None:
                raw_content = file_path.read_bytes()
            if b"\x00" in raw_content:
                raise ValueError(f"could not read document {file_path}: binary content")
            text = raw_content.decode("utf-8")
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"could not read document {file_path}: {exc}") from exc
        return self.chunk_text(
            text, file_path.name, source_file=file_path.name, corpus_hash=corpus_hash
        )

    def config_dict(self) -> dict[str, int]:
        """Return a JSON-serializable configuration."""
        return {
            "chunk_size": self.config.chunk_size,
            "character_overlap": self.config.character_overlap,
            "token_overlap": self.config.token_overlap,
        }
