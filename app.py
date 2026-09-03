"""Streamlit interface for interactive grounded conversations."""

import os
from pathlib import Path
from typing import Any, cast

import httpx
import streamlit as st

from config import Settings
from core.models import RAGQueryRequest
from core.service import RAGService
from factory import get_rag_service
from ingestion import ingest_directory


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


def main() -> None:
    """Render the RAG chat application."""
    st.set_page_config(page_title="Enterprise Hybrid RAG", page_icon="R", layout="wide")
    st.session_state.setdefault("messages", [])
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
        st.caption(
            f"Embedding: {cast(str, metadata.get('embedding_model', 'unknown'))}"
        )
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
                st.toast(f"Indexed {count} chunks", icon="OK")
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
    if prompt := st.chat_input("Ask a question about your documents"):
        messages = st.session_state.setdefault("messages", [])
        messages.append({"role": "user", "answer": prompt})
        del messages[:-30]
        try:
            with st.spinner("Thinking & retrieving sources..."):
                if api_mode:
                    response_data = _api_request(
                        "POST",
                        "/query",
                        json={"query": prompt, "temperature": 0.0, "confidence_threshold": threshold},
                    )
                    answer = cast(str, response_data["answer"])
                    sources = cast(list[dict[str, object]], response_data.get("sources", []))
                    latency_seconds = cast(float, response_data.get("latency_seconds", 0.0))
                else:
                    if service is None:
                        raise RuntimeError("direct service is unavailable")
                    response = service.query(
                        RAGQueryRequest(query=prompt, temperature=0.0, confidence_threshold=threshold)
                    )
                    answer = response.answer
                    sources = [item.model_dump() for item in response.sources]
                    latency_seconds = response.latency_seconds
            messages.append(
                {
                    "role": "assistant",
                    "answer": answer,
                    "sources": sources,
                    "latency_seconds": latency_seconds,
                }
            )
            del messages[:-30]
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
