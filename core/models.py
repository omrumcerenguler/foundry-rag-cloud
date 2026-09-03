"""Pydantic schemas exchanged by the RAG application layer."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentChunk(BaseModel):
    """A searchable chunk of a source document."""

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    embedding: list[float] | None = None
    source_file: str | None = None
    character_offset: int = Field(default=0, ge=0)
    corpus_hash: str | None = None


class SearchResult(BaseModel):
    """A document chunk returned by vector similarity search."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    chunk_index: int = Field(ge=0)
    text: str
    score: float = Field(ge=-1.0, le=1.0)
    source_file: str | None = None
    character_offset: int | None = None
    corpus_hash: str | None = None


class RAGQueryRequest(BaseModel):
    """User query and retrieval parameters for a RAG request."""

    query: str = Field(min_length=1, max_length=2000)
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_k: int = Field(default=5, ge=1, le=20)
    confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("query", mode="before")
    @classmethod
    def validate_query(cls, value: object) -> object:
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ValueError("query must be non-empty and must not contain null bytes")
        return value


class RAGResponse(BaseModel):
    """Grounded answer, provenance, and execution telemetry."""

    answer: str
    sources: list[SearchResult]
    citations: list[str]
    latency_seconds: float = Field(ge=0.0)
    mode: str
