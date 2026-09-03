"""Production HTTP API for the hybrid RAG service."""

import os
import secrets
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from fastapi import FastAPI, Request, Security
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.middleware.cors import CORSMiddleware

from config import Settings
from core.models import RAGQueryRequest
from core.service import RAGService
from factory import get_rag_service
from ingestion import ingest_directory
from providers.azure_openai import (
    AzureAuthError,
    AzureAuthorizationError,
    AzureRateLimitError,
)


class QueryRequest(BaseModel):
    """Parameters accepted by the query endpoint."""

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=20)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("query", mode="before")
    @classmethod
    def validate_query(cls, value: object) -> object:
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ValueError("query must be non-empty and must not contain null bytes")
        return value


class QueryResponse(BaseModel):
    """Grounded answer and retrieval telemetry."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[str]
    confidence_score: float
    mode: str
    latency_seconds: float
    sources: list[dict[str, Any]]


class IngestResponse(BaseModel):
    """Summary of an atomic index replacement."""

    chunks_indexed: int
    duration_seconds: float
    corpus_hash: str
    model_name: str


class HealthResponse(BaseModel):
    """Readiness information for orchestration probes."""

    status: str
    mode: str
    total_chunks: int
    embedding_model: str


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _service(request: Request) -> RAGService:
    """Get the one service instance created by the application lifespan."""
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise RuntimeError("RAG service is not initialized")
    return cast(RAGService, service)


def _metadata(service: RAGService) -> dict[str, object]:
    """Read store metadata while tolerating an empty index."""
    metadata = service.vector_store.get_index_metadata()
    return metadata or {
        "embedding_model": service.embedding_provider.model_name,
        "vector_dimension": service.embedding_provider.dimension,
        "total_chunk_count": 0,
        "chunking_config": {},
        "ingestion_timestamp": None,
    }


def _error(status_code: int, detail: str) -> JSONResponse:
    """Build the stable error envelope used by all exception handlers."""
    return JSONResponse(status_code=status_code, content={"error": {"detail": detail}})


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Create one service per process and close its store on shutdown."""
    service: RAGService | None = None
    startup_error = ""
    settings = Settings.from_env()
    application.state.settings = settings
    try:
        service = get_rag_service(settings)
    except Exception:
        startup_error = "RAG service initialization failed"
    application.state.service = service
    application.state.startup_error = startup_error
    try:
        yield
    finally:
        if service is not None:
            service.vector_store.close()


app = FastAPI(title="Enterprise Hybrid RAG API", version="3.0.0", lifespan=lifespan)
allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key"],
    )


@app.middleware("http")
async def api_key_middleware(request: Request, call_next: Any) -> Any:
    """Reject protected requests before FastAPI reads or validates their body."""
    if request.url.path in {"/query", "/ingest", "/metadata"}:
        settings = getattr(request.app.state, "settings", Settings.from_env())
        expected = settings.api_key or ""
        supplied = request.headers.get("X-API-Key")
        if bool(settings.api_key and settings.api_key.strip()) and (
            supplied is None or not secrets.compare_digest(supplied, expected)
        ):
            return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    _: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return validation failures in the API error envelope."""
    errors = [
        {key: value for key, value in error.items() if key != "ctx"}
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"error": {"detail": errors}})


@app.exception_handler(AzureAuthError)
async def azure_auth_exception_handler(_: Request, exc: AzureAuthError) -> JSONResponse:
    """Map rejected Azure credentials to HTTP 401."""
    return _error(401, str(exc))


@app.exception_handler(AzureRateLimitError)
async def azure_rate_limit_exception_handler(
    _: Request, exc: AzureRateLimitError
) -> JSONResponse:
    """Map Azure throttling to HTTP 429."""
    return _error(429, str(exc))


@app.exception_handler(AzureAuthorizationError)
async def azure_authorization_exception_handler(
    _: Request, exc: AzureAuthorizationError
) -> JSONResponse:
    """Map insufficient Azure permissions to HTTP 403."""
    return _error(403, str(exc))


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, _exc: ValueError) -> JSONResponse:
    """Map domain validation errors to a sanitized HTTP 400 response."""
    return _error(400, "Invalid request")


@app.exception_handler(RuntimeError)
async def runtime_error_handler(_: Request, _exc: RuntimeError) -> JSONResponse:
    """Map service and store failures to a sanitized HTTP 503 response."""
    return _error(503, "Service unavailable")


@app.exception_handler(Exception)
async def unexpected_exception_handler(_: Request, _exc: Exception) -> JSONResponse:
    """Return a stable, non-sensitive response for unexpected failures."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "detail": "An unexpected internal error occurred.",
        },
    )


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Return readiness and current index statistics."""
    service = getattr(request.app.state, "service", None)
    if service is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "mode": "UNKNOWN",
                "total_chunks": 0,
                "embedding_model": "unavailable",
            },
        )  # type: ignore[return-value]
    store = service.vector_store
    database_path = getattr(store, "database_path", ":memory:")
    healthy = database_path == ":memory:" or Path(str(database_path)).exists()
    try:
        health_check = getattr(store, "check_health", None)
        if callable(health_check):
            healthy = healthy and bool(health_check())
    except Exception:
        healthy = False
    try:
        metadata = _metadata(service)
    except Exception:
        metadata = {
            "embedding_model": "unavailable",
            "vector_dimension": 0,
            "total_chunk_count": 0,
        }
        healthy = False
    total_chunks = cast(int, metadata.get("total_chunk_count", 0))
    status = "ready" if healthy and total_chunks > 0 else "degraded"
    response = HealthResponse(
        status=status,
        mode=service.mode,
        total_chunks=total_chunks,
        embedding_model=cast(str, metadata.get("embedding_model", "unavailable")),
    )
    if status == "degraded":
        return JSONResponse(status_code=503, content=response.model_dump())  # type: ignore[return-value]
    return response


def _require_api_key(
    request: Request, api_key: str | None = Security(api_key_header)
) -> None:
    """Require a matching key only when API_KEY protection is configured."""
    settings = getattr(request.app.state, "settings", Settings.from_env())
    expected = settings.api_key or ""
    supplied = api_key or ""
    if bool(settings.api_key and settings.api_key.strip()) and (
        api_key is None or not secrets.compare_digest(supplied, expected)
    ):
        raise PermissionError("invalid API key")


@app.post("/query", response_model=QueryResponse)
def query(
    payload: QueryRequest,
    request: Request,
    _: None = Security(_require_api_key),
) -> QueryResponse:
    """Execute a synchronous RAG query in FastAPI's worker thread."""
    response = _service(request).query(
        RAGQueryRequest(
            query=payload.query,
            top_k=payload.top_k,
            temperature=payload.temperature,
            confidence_threshold=payload.confidence_threshold,
        )
    )
    sources = [source.model_dump() for source in response.sources]
    confidence = max((source.score for source in response.sources), default=0.0)
    return QueryResponse(
        answer=response.answer,
        citations=response.citations,
        confidence_score=confidence,
        mode=response.mode,
        latency_seconds=response.latency_seconds,
        sources=sources,
    )


@app.exception_handler(PermissionError)
async def permission_exception_handler(
    _: Request, exc: PermissionError
) -> JSONResponse:
    """Map failed API key authentication to HTTP 401."""
    return _error(401, str(exc))


@app.post("/ingest", response_model=IngestResponse)
def ingest(
    request: Request,
    _: None = Security(_require_api_key),
) -> IngestResponse:
    """Atomically ingest the configured data directory."""
    service = _service(request)
    data_dir = Path(__file__).parent.resolve() / "data"
    configured_dir = Path(os.getenv("RAG_DATA_DIR", str(data_dir)))
    if not configured_dir.is_absolute():
        configured_dir = Path(__file__).parent.resolve() / configured_dir
    started = perf_counter()
    corpus_hash: list[str] = []
    count = ingest_directory(
        configured_dir,
        service.embedding_provider,
        service.vector_store,
        corpus_hash_out=corpus_hash,
    )
    metadata = _metadata(service)
    return IngestResponse(
        chunks_indexed=count,
        duration_seconds=perf_counter() - started,
        corpus_hash=corpus_hash[0] if corpus_hash else "",
        model_name=cast(str, metadata["embedding_model"]),
    )


@app.get("/metadata")
def metadata(
    request: Request, _: None = Security(_require_api_key)
) -> dict[str, object]:
    """Return index parameters and vector statistics."""
    return _metadata(_service(request))
