from pathlib import Path

import pytest

from core.chunking import DocumentChunker


def test_chunk_boundaries_and_metadata() -> None:
    chunks = DocumentChunker(chunk_size=5, character_overlap=2).chunk_text("abcdefghij", "doc")
    assert [chunk.text for chunk in chunks] == ["abcde", "defgh", "ghij"]
    assert [chunk.character_offset for chunk in chunks] == [0, 3, 6]


def test_empty_and_corrupted_files(tmp_path: Path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text(" \n", encoding="utf-8")
    assert DocumentChunker().chunk_file(empty) == []
    corrupted = tmp_path / "bad.txt"
    corrupted.write_bytes(b"\xff\xfe")
    with pytest.raises(ValueError, match="could not read"):
        DocumentChunker().chunk_file(corrupted)