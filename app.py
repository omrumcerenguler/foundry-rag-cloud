"""Streamlit interface for interactive grounded conversations."""

import os
from pathlib import Path
from time import monotonic
from typing import Any, cast

import httpx
import streamlit as st

from config import Settings
from core.models import RAGQueryRequest
from core.service import RAGService
from factory import get_rag_service
from ingestion import ingest_directory

KNOWLEDGE_BASE_DOCUMENTS = (
    (
        "doc1.txt",
        "Local AI development",
        "Offline inference keeps prompts and documents local, removes network dependency, and supports repeatable development.",
    ),
    (
        "doc2.txt",
        "Python environments",
        "Virtual environments isolate project dependencies from the system interpreter.",
    ),
    (
        "doc3.txt",
        "RAG ingestion",
        "RAG ingestion reads documents, creates passages and embeddings, and stores vectors for later search.",
    ),
    (
        "doc4.txt",
        "SQLite knowledge bases",
        "SQLite stores text and serialized embeddings while Python calculates cosine similarity.",
    ),
    (
        "doc5.txt",
        "Apple Silicon compatibility",
        "ARM64 Python, compatible macOS wheels, and reproducible environments support local model execution.",
    ),
    (
        "project_plan.txt",
        "Project delivery plan",
        "A six-week plan covers foundations, implementation, testing, documentation, and the final demo.",
    ),
)

SUGGESTED_QUESTIONS = (
    "Why is local AI development useful for privacy and repeatable development?",
    "How does a Python virtual environment isolate project dependencies?",
    "What steps does the RAG ingestion pipeline perform before search?",
    "How does SQLite support a small local vector knowledge base?",
    "What setup is recommended for local AI development on Apple Silicon?",
    "What are the three phases of the local RAG AI assistant project plan?",
)

MAX_SESSION_QUERIES = 8
QUERY_COOLDOWN_SECONDS = 4.0
QueryResult = tuple[str, list[dict[str, object]], float]


def _friendly_error(prefix: str) -> str:
    """Return a stable UI error without exposing internal exception details."""
    return prefix


@st.cache_resource
def get_service() -> RAGService:
    """Create and cache one service for the Streamlit process."""
    return get_rag_service(Settings.from_env())


def _api_headers() -> dict[str, str]:
    """Return authentication headers for the configured API service."""
    key = os.getenv("API_KEY", "")
    return {"X-API-Key": key} if key.strip() else {}


def _api_request(method: str, path: str, **kwargs: Any) -> dict[str, object]:
    """Call the API service and return its JSON object response."""
    base_url = os.environ["RAG_API_URL"].rstrip("/")
    try:
        response = httpx.request(
            method, f"{base_url}{path}", headers=_api_headers(), timeout=30.0, **kwargs
        )
    except httpx.TimeoutException as exc:
        raise RuntimeError("backend request timed out") from exc
    except httpx.RequestError as exc:
        raise RuntimeError("backend connection failed") from exc
    if response.status_code == 429:
        raise RuntimeError("backend rate limit reached")
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("API returned malformed data")
    return cast(dict[str, object], payload)


def _query_service(
    service: RAGService | None,
    api_mode: bool,
    prompt: str,
    threshold: float,
) -> tuple[str, list[dict[str, object]], float]:
    """Run one query through the configured API or direct service path."""
    with st.spinner("Thinking & retrieving sources..."):
        if api_mode:
            response_data = _api_request(
                "POST",
                "/query",
                json={
                    "query": prompt,
                    "temperature": 0.0,
                    "confidence_threshold": threshold,
                },
            )
            return (
                cast(str, response_data["answer"]),
                cast(list[dict[str, object]], response_data.get("sources", [])),
                cast(float, response_data.get("latency_seconds", 0.0)),
            )
        if service is None:
            raise RuntimeError("direct service is unavailable")
        response = service.query(
            RAGQueryRequest(
                query=prompt, temperature=0.0, confidence_threshold=threshold
            )
        )
        return (
            response.answer,
            [item.model_dump() for item in response.sources],
            response.latency_seconds,
        )


def _append_query_messages(
    prompt: str,
    messages: list[dict[str, object]],
    answer: str,
    sources: list[dict[str, object]],
    latency_seconds: float,
) -> None:
    """Append a user query and its grounded response within the history bound."""
    messages.append({"role": "user", "answer": prompt})
    messages.append(
        {
            "role": "assistant",
            "answer": answer,
            "sources": sources,
            "latency_seconds": latency_seconds,
        }
    )
    del messages[:-30]


def _query_cache_key(prompt: str, threshold: float) -> str:
    """Build a stable cache key for an equivalent prompt configuration."""
    return f"{prompt.strip().casefold()}::{threshold:.4f}"


def _consume_query_slot() -> bool:
    """Enforce per-session quota and cooldown before an external query."""
    now = monotonic()
    query_count = int(st.session_state.get("query_count", 0))
    if query_count >= MAX_SESSION_QUERIES:
        st.warning(
            f"Session limit reached ({MAX_SESSION_QUERIES} questions). Please return later."
        )
        return False
    last_query_at = float(st.session_state.get("last_query_at", 0.0))
    elapsed = now - last_query_at
    if last_query_at and elapsed < QUERY_COOLDOWN_SECONDS:
        remaining = QUERY_COOLDOWN_SECONDS - elapsed
        st.warning(f"Please wait {remaining:.1f} seconds before asking another question.")
        return False
    st.session_state["query_count"] = query_count + 1
    st.session_state["last_query_at"] = now
    return True


def main() -> None:
    """Render the RAG chat application."""
    st.set_page_config(page_title="Enterprise Hybrid RAG", page_icon="R", layout="wide")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("query_count", 0)
    st.session_state.setdefault("last_query_at", 0.0)
    st.session_state.setdefault("query_cache", {})
    api_mode = bool(os.getenv("RAG_API_URL", "").strip())
    try:
        service = None if api_mode else get_service()
        settings = Settings.from_env()
        if api_mode:
            metadata = _api_request("GET", "/metadata")
        else:
            if service is None:
                raise RuntimeError("direct service is unavailable")
            metadata = service.vector_store.get_index_metadata() or {}
        if not isinstance(metadata, dict):
            raise RuntimeError("vector store metadata is malformed")
    except Exception:
        st.error(_friendly_error("RAG service is unavailable. Check the backend configuration."))
        return
    with st.sidebar:
        st.title("RAG Control")
        mode = "API"
        if service is not None:
            mode = service.mode
        st.success(f"ACTIVE · {mode}")
        st.metric("Indexed chunks", cast(int, metadata.get("total_chunk_count", 0)))
        quota_used = min(
            int(st.session_state["query_count"]), MAX_SESSION_QUERIES
        )
        quota_remaining = MAX_SESSION_QUERIES - quota_used
        st.metric(
            "Session Quota",
            f"{quota_remaining} / {MAX_SESSION_QUERIES} Remaining",
            help="This per-session allowance protects the public demo's Azure credits while keeping the demo available for genuine visitors.",
        )
        st.progress(
            quota_used / MAX_SESSION_QUERIES,
            text=f"{quota_used} of {MAX_SESSION_QUERIES} questions used",
        )
        if quota_used >= MAX_SESSION_QUERIES:
            st.error("Demo session limit reached. Please return later to try again.")
        elif quota_used >= MAX_SESSION_QUERIES - 1:
            st.warning(
                f"Almost at the demo limit: {quota_remaining} question remaining."
            )
        st.caption(
            f"Embedding: {cast(str, metadata.get('embedding_model', 'unknown'))}"
        )
        with st.expander("📚 Knowledge Base", expanded=True):
            for document_name, topic, description in KNOWLEDGE_BASE_DOCUMENTS:
                st.markdown(f"**{document_name}** · {topic}")
                st.caption(description)
        threshold = st.slider(
            "Confidence threshold", 0.0, 1.0, settings.confidence_threshold, 0.01
        )
        if st.button("Ingest Documents", type="primary", use_container_width=True):
            try:
                data_dir = Path(os.getenv("RAG_DATA_DIR", "data"))
                if not data_dir.is_absolute():
                    data_dir = Path(__file__).parent.resolve() / data_dir
                if api_mode:
                    count = cast(int, _api_request("POST", "/ingest")["chunks_indexed"])
                else:
                    if service is None:
                        raise RuntimeError("direct service is unavailable")
                    count = ingest_directory(data_dir, service.embedding_provider, service.vector_store)
                st.toast(f"Indexed {count} chunks", icon="✅")
                st.rerun()
            except (RuntimeError, ValueError, OSError):
                st.error(_friendly_error("Document ingestion failed. Check the data directory and service status."))
    st.title("Grounded Knowledge Assistant")
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["answer"])
            if message.get("sources"):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.caption(
                            f"{source['source_id']} · score {source['score']:.3f}"
                        )
                        st.write(source["text"])
                    st.caption(f"Latency: {message['latency_seconds']:.3f}s")

    st.subheader("💡 Suggested Questions")
    question_columns = st.columns(2)
    for index, suggested_question in enumerate(SUGGESTED_QUESTIONS):
        if question_columns[index % 2].button(
            suggested_question,
            key=f"suggested-question-{index}",
            use_container_width=True,
        ):
            st.session_state["pending_prompt"] = suggested_question

    prompt = st.session_state.pop("pending_prompt", None)
    if prompt is None:
        prompt = st.chat_input("Ask a question about your documents")
    if prompt and _consume_query_slot():
        messages = st.session_state.setdefault("messages", [])
        try:
            cache_key = _query_cache_key(prompt, threshold)
            query_cache = cast(dict[str, QueryResult], st.session_state["query_cache"])
            if cache_key in query_cache:
                answer, sources, latency_seconds = query_cache[cache_key]
            else:
                answer, sources, latency_seconds = _query_service(
                    service, api_mode, prompt, threshold
                )
                query_cache[cache_key] = (answer, sources, latency_seconds)
            _append_query_messages(
                prompt, messages, answer, sources, latency_seconds
            )
        except RuntimeError as exc:
            error_message = str(exc)
            if error_message == "backend rate limit reached":
                message = "The backend is rate-limited. Please retry in a moment."
            elif error_message == "backend request timed out":
                message = "The backend took too long to respond. Please retry."
            elif error_message == "backend connection failed":
                message = "The backend connection was lost. Check service status and retry."
            else:
                message = "Query failed. The service may be unavailable or the index may be empty."
            st.error(_friendly_error(message))
        except Exception:
            st.error(_friendly_error("Query failed. The service may be unavailable or the index may be empty."))
        else:
            st.rerun()


if __name__ == "__main__":
    main()
