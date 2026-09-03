"""Streamlit interface for interactive grounded conversations."""

import html
import json
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

KnowledgeBaseDocument = tuple[str, str, str, str, tuple[str, ...]]
KNOWLEDGE_BASE_DOCUMENTS: tuple[KnowledgeBaseDocument, ...] = (
    (
        "doc1.txt",
        "Local AI development",
        "Offline inference keeps prompts and documents local, removes network dependency, and supports repeatable development.",
        "Privacy-preserving local inference and reproducible AI workflows.",
        ("offline inference", "data privacy", "network independence", "reproducibility"),
    ),
    (
        "doc2.txt",
        "Python environments",
        "Virtual environments isolate project dependencies from the system interpreter.",
        "Dependency isolation for predictable Python project setup.",
        ("venv", "dependency isolation", "pip", "reproducible setup"),
    ),
    (
        "doc3.txt",
        "RAG ingestion",
        "RAG ingestion reads documents, creates passages and embeddings, and stores vectors for later search.",
        "The ingestion path from source documents to searchable vector representations.",
        ("chunking", "embeddings", "vector search", "document pipeline"),
    ),
    (
        "doc4.txt",
        "SQLite knowledge bases",
        "SQLite stores text and serialized embeddings while Python calculates cosine similarity.",
        "A lightweight embedded vector store design for small local knowledge bases.",
        ("SQLite", "BLOB storage", "cosine similarity", "local persistence"),
    ),
    (
        "doc5.txt",
        "Apple Silicon compatibility",
        "ARM64 Python, compatible macOS wheels, and reproducible environments support local model execution.",
        "Runtime and package choices for reliable local AI on Apple Silicon.",
        ("ARM64", "macOS wheels", "Foundry Local", "hardware compatibility"),
    ),
    (
        "project_plan.txt",
        "Project delivery plan",
        "A six-week plan covers foundations, implementation, testing, documentation, and the final demo.",
        "A phased six-week delivery plan for a Local RAG AI Assistant.",
        ("Foundations", "Build Project", "Testing", "documentation", "demo"),
    ),
)

QUESTION_LIBRARY: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "🏗️ Architecture & Phases",
        (
            "How does SQLite store embeddings and support cosine similarity search?",
            "What are the three phases and week ranges in the project delivery plan?",
            "How does the RAG assistant separate ingestion, storage, retrieval, and generation?",
            "Why is SQLite an appropriate vector store for a small local knowledge base?",
            "What deliverables are planned for the Testing & Wrap-up phase?",
        ),
    ),
    (
        "⚙️ Data Ingestion & RAG Pipeline",
        (
            "Why is local AI useful when prompts and documents must remain private?",
            "What steps transform source documents into searchable RAG vectors?",
            "How do chunking and embeddings prepare passages for semantic retrieval?",
            "What network dependency is removed by running inference locally?",
            "How does the ingestion pipeline connect source text to later search?",
        ),
    ),
    (
        "💻 Runtime & Hardware Optimization",
        (
            "How does a Python virtual environment isolate project dependencies?",
            "Why should Apple Silicon users choose ARM64 Python and compatible wheels?",
            "How does a virtual environment improve reproducibility across machines?",
            "What package compatibility concerns matter for local AI on macOS?",
            "How do Python environments and Apple Silicon compatibility work together for local models?",
        ),
    ),
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
    with st.status(
        "🔍 Retrieving relevant chunks from SQLite Vector Store...", expanded=False
    ) as status:
        try:
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
                result = (
                    cast(str, response_data["answer"]),
                    cast(list[dict[str, object]], response_data.get("sources", [])),
                    cast(float, response_data.get("latency_seconds", 0.0)),
                )
            else:
                if service is None:
                    raise RuntimeError("direct service is unavailable")
                response = service.query(
                    RAGQueryRequest(
                        query=prompt, temperature=0.0, confidence_threshold=threshold
                    )
                )
                result = (
                    response.answer,
                    [item.model_dump() for item in response.sources],
                    response.latency_seconds,
                )
            status.update(
                label="⚡ Synthesizing grounded answer via Azure OpenAI...",
                state="complete",
            )
            return result
        except Exception:
            status.update(label="Query failed", state="error")
            raise


def _append_query_messages(
    prompt: str,
    messages: list[dict[str, object]],
    answer: str,
    sources: list[dict[str, object]],
    latency_seconds: float,
    observability: dict[str, object] | None = None,
) -> None:
    """Append a user query and its grounded response within the history bound."""
    messages.append({"role": "user", "answer": prompt})
    assistant_message: dict[str, object] = {
        "role": "assistant",
        "answer": answer,
        "sources": sources,
        "latency_seconds": latency_seconds,
    }
    if observability is not None:
        assistant_message["observability"] = observability
    messages.append(assistant_message)
    del messages[:-30]


def _float_value(value: object, default: float = 0.0) -> float:
    """Read a numeric metadata value without trusting persisted session data."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _int_value(value: object, default: int = 0) -> int:
    """Read an integer metadata value without trusting persisted session data."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _conversation_markdown(messages: list[dict[str, object]]) -> str:
    """Serialize the current conversation into a portable Markdown report."""
    sections = ["# Grounded RAG Conversation", ""]
    for message in messages:
        role = str(message.get("role", "message")).title()
        sections.append(f"## {role}")
        sections.append(str(message.get("answer", "")))
        if role == "Assistant" and message.get("sources"):
            sections.append("\n**Sources:**")
            for source in cast(list[dict[str, object]], message["sources"]):
                sections.append(
                    f"- {source.get('source_id', 'unknown')} (score: {_float_value(source.get('score')):.3f})"
                )
        sections.append("")
    return "\n".join(sections)


def _render_copy_button(answer: str) -> None:
    """Render a browser clipboard action for one assistant response."""
    encoded_answer = html.escape(json.dumps(answer), quote=True)
    st.markdown(
        f'<button class="rag-copy-button" type="button" onclick="navigator.clipboard.writeText({encoded_answer})">Copy answer</button>',
        unsafe_allow_html=True,
    )


def _render_response_observability(message: dict[str, object]) -> None:
    """Render compact telemetry pills for an assistant response."""
    telemetry = cast(dict[str, object], message.get("observability", {}))
    latency_ms = _float_value(telemetry.get("latency_ms"))
    confidence = _float_value(telemetry.get("confidence"))
    match_count = _int_value(telemetry.get("match_count"))
    model = str(telemetry.get("model", "unknown"))
    cache = str(telemetry.get("cache", "MISS"))
    st.markdown(
        "<div class=\"rag-observability\">"
        f"<span>Latency {latency_ms:.0f} ms</span>"
        f"<span>Confidence {confidence:.3f}</span>"
        f"<span>{match_count} match{'es' if match_count != 1 else ''}</span>"
        f"<span>{html.escape(model)}</span>"
        f"<span class=\"rag-cache-{cache.lower()}\">Cache {html.escape(cache)}</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    _render_copy_button(str(message.get("answer", "")))


def _render_citation_inspector(sources: list[dict[str, object]]) -> None:
    """Render expandable source passages with visual relevance strength."""
    for index, source in enumerate(sources):
        score = max(0.0, min(1.0, _float_value(source.get("score"))))
        with st.expander(
            f"Citation {index + 1} · {source.get('source_id', 'unknown')} · {score:.3f}"
        ):
            st.markdown(f"**Cosine relevance:** {score:.3f}")
            st.progress(score, text=f"Relevance strength · {score:.0%}")
            st.write(source.get("text", ""))


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
    st.markdown(
        """
        <style>
        :root {
            --rag-ink: #f3f7ff;
            --rag-muted: #9baac0;
            --rag-panel: rgba(17, 25, 39, 0.72);
            --rag-panel-strong: rgba(22, 31, 49, 0.92);
            --rag-line: rgba(255, 255, 255, 0.11);
            --rag-cyan: #67e8f9;
            --rag-indigo: #818cf8;
            --rag-violet: #a78bfa;
            --rag-accent-soft: rgba(99, 102, 241, 0.14);
        }
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 88% 8%, rgba(99, 102, 241, 0.13), transparent 28rem),
                radial-gradient(circle at 12% 92%, rgba(34, 211, 238, 0.08), transparent 24rem),
                #080d16;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(11, 17, 29, 0.97), rgba(10, 14, 24, 0.93));
            border-right: 1px solid var(--rag-line);
        }
        .rag-hero {
            padding: 1.35rem 0 1rem;
            border-bottom: 1px solid rgba(129, 140, 248, 0.2);
            margin-bottom: 1.1rem;
        }
        .rag-hero h1 {
            background: linear-gradient(135deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            font-size: 3rem;
            letter-spacing: 0;
            line-height: 1.08;
            margin: 0;
        }
        .rag-hero p {
            color: var(--rag-muted);
            font-size: 1.02rem;
            letter-spacing: 0;
            margin: 0.55rem 0 0;
        }
        .rag-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.9rem;
        }
        .rag-badge {
            background: linear-gradient(135deg, rgba(103, 232, 249, 0.11), rgba(167, 139, 250, 0.11));
            border: 1px solid rgba(129, 140, 248, 0.3);
            border-radius: 999px;
            color: #d6e4ff;
            font-size: 0.76rem;
            font-weight: 600;
            padding: 0.35rem 0.65rem;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06);
        }
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(20, 31, 49, 0.88), rgba(14, 21, 35, 0.78));
            border: 1px solid rgba(103, 232, 249, 0.22);
            border-radius: 9px;
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.18), inset 0 1px 0 rgba(255, 255, 255, 0.05);
            min-width: 0;
            overflow: visible;
            padding: 0.8rem 0.85rem;
        }
        [data-testid="stMetricLabel"] {
            color: var(--rag-muted);
        }
        [data-testid="stMetricValue"] {
            color: var(--rag-ink);
            font-size: clamp(1rem, 2.1vw, 1.55rem);
            white-space: nowrap;
        }
        [data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.025);
            border-color: var(--rag-line);
            border-radius: 9px;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.025);
        }
        [data-testid="stExpanderDetails"] {
            background: rgba(12, 19, 32, 0.58);
        }
        .rag-status {
            align-items: center;
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.18), rgba(6, 78, 59, 0.3));
            border: 1px solid rgba(52, 211, 153, 0.38);
            border-radius: 999px;
            color: #a7f3d0;
            display: inline-flex;
            font-size: 0.76rem;
            font-weight: 700;
            gap: 0.45rem;
            letter-spacing: 0.04em;
            padding: 0.42rem 0.7rem;
        }
        .rag-status::before {
            background: #34d399;
            border-radius: 50%;
            box-shadow: 0 0 0 3px rgba(52, 211, 153, 0.14), 0 0 13px rgba(52, 211, 153, 0.85);
            content: "";
            height: 0.48rem;
            width: 0.48rem;
        }
        .rag-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 0.3rem;
            margin-top: 0.45rem;
        }
        .rag-tag {
            background: rgba(129, 140, 248, 0.12);
            border: 1px solid rgba(129, 140, 248, 0.28);
            border-radius: 999px;
            color: #c7d2fe;
            font-size: 0.68rem;
            padding: 0.2rem 0.45rem;
        }
        .rag-observability {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            margin: 0.55rem 0 0.2rem;
        }
        .rag-observability span {
            background: rgba(103, 232, 249, 0.08);
            border: 1px solid rgba(103, 232, 249, 0.2);
            border-radius: 999px;
            color: #b9c8dd;
            font-size: 0.69rem;
            padding: 0.24rem 0.5rem;
        }
        .rag-observability .rag-cache-hit {
            background: rgba(52, 211, 153, 0.1);
            border-color: rgba(52, 211, 153, 0.3);
            color: #a7f3d0;
        }
        .rag-copy-button {
            background: transparent;
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 7px;
            color: var(--rag-muted);
            cursor: pointer;
            font-size: 0.7rem;
            padding: 0.25rem 0.55rem;
            transition: color 160ms ease, border-color 160ms ease;
        }
        .rag-copy-button:hover {
            border-color: var(--rag-cyan);
            color: var(--rag-ink);
        }
        [data-baseweb="tab-list"] {
            gap: 0.35rem;
            border-bottom: 1px solid rgba(129, 140, 248, 0.2);
        }
        [data-baseweb="tab"] {
            color: var(--rag-muted);
            min-height: 3rem;
            padding: 0.6rem 0.85rem;
        }
        [aria-selected="true"][data-baseweb="tab"] {
            color: var(--rag-ink);
        }
        [data-baseweb="tab-highlight"] {
            background: linear-gradient(90deg, var(--rag-cyan), var(--rag-indigo));
            height: 3px;
        }
        .stButton > button {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 9px;
            color: var(--rag-ink);
            min-height: 3.8rem;
            transition: background-color 180ms ease, border-color 180ms ease, box-shadow 180ms ease, transform 180ms ease;
        }
        .stButton > button:hover {
            background: rgba(99, 102, 241, 0.12);
            border-color: rgba(129, 140, 248, 0.72);
            box-shadow: 0 4px 20px rgba(99, 102, 241, 0.2);
            color: #ffffff;
            transform: translateY(-2px);
        }
        .rag-empty-state {
            background: linear-gradient(135deg, rgba(20, 31, 49, 0.84), rgba(42, 31, 72, 0.54));
            border: 1px solid rgba(129, 140, 248, 0.28);
            border-left: 3px solid var(--rag-cyan);
            border-radius: 9px;
            box-shadow: 0 16px 36px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.06);
            color: #b4c1d4;
            margin: 0.4rem 0 1.25rem;
            padding: 1rem 1.15rem;
        }
        .rag-empty-state strong {
            color: var(--rag-ink);
            display: block;
            font-size: 1rem;
            margin-bottom: 0.25rem;
        }
        @media (max-width: 640px) {
            .rag-hero h1 {
                font-size: 2rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
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
        st.markdown(
            f'<div class="rag-status">ACTIVE · {mode}</div>',
            unsafe_allow_html=True,
        )
        st.link_button(
            "GitHub Repository",
            "https://github.com/omrumcerenguler/foundry-rag-cloud",
            use_container_width=True,
        )
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()
        st.download_button(
            "Download Conversation / Report",
            data=_conversation_markdown(
                cast(list[dict[str, object]], st.session_state["messages"])
            ),
            file_name="grounded-rag-conversation.md",
            mime="text/markdown",
            use_container_width=True,
        )
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
            for (
                document_name,
                domain,
                summary,
                scope,
                key_concepts,
            ) in KNOWLEDGE_BASE_DOCUMENTS:
                with st.expander(document_name):
                    st.markdown(f"**Domain:** {domain}")
                    st.markdown(f"**Scope:** {scope}")
                    st.markdown(f"**Summary:** {summary}")
                    st.markdown(
                        "**Key Concepts:** "
                        + '<div class="rag-tags">'
                        + "".join(
                            f'<span class="rag-tag">{concept}</span>'
                            for concept in key_concepts
                        )
                        + "</div>",
                        unsafe_allow_html=True,
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
                st.toast(f"Indexed {count} chunks", icon="✅")
                st.rerun()
            except (RuntimeError, ValueError, OSError):
                st.error(_friendly_error("Document ingestion failed. Check the data directory and service status."))
    st.markdown(
        """
        <section class="rag-hero">
            <h1>Grounded Knowledge Assistant</h1>
            <p>Enterprise Grounded RAG with Hybrid Search &amp; Deterministic Citations</p>
            <div class="rag-badges">
                <span class="rag-badge">Azure OpenAI · gpt-4.1-mini</span>
                <span class="rag-badge">SQLite Vector</span>
                <span class="rag-badge">Session Rate-Limited</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    messages = st.session_state.get("messages", [])
    if not messages:
        st.markdown(
            """
            <div class="rag-empty-state">
                <strong>Ready when you are</strong>
                Choose a suggested question below or ask about the indexed knowledge base.
                Answers include grounded source citations when the documents support them.
            </div>
            """,
            unsafe_allow_html=True,
        )
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["answer"])
            if message.get("sources"):
                _render_citation_inspector(
                    cast(list[dict[str, object]], message["sources"])
                )
            if message.get("observability"):
                _render_response_observability(message)

    st.subheader("💡 Suggested Questions")
    question_tabs = st.tabs([category for category, _ in QUESTION_LIBRARY])
    for tab, (_, questions) in zip(question_tabs, QUESTION_LIBRARY):
        with tab:
            question_columns = st.columns(2)
            for index, suggested_question in enumerate(questions):
                if question_columns[index % 2].button(
                    suggested_question,
                    key=f"suggested-question-{index}-{questions[0][:12]}",
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
            cache_hit = cache_key in query_cache
            if cache_hit:
                answer, sources, latency_seconds = query_cache[cache_key]
            else:
                answer, sources, latency_seconds = _query_service(
                    service, api_mode, prompt, threshold
                )
                query_cache[cache_key] = (answer, sources, latency_seconds)
            _append_query_messages(
                prompt,
                messages,
                answer,
                sources,
                latency_seconds,
                {
                    "latency_ms": latency_seconds * 1000,
                    "confidence": max(
                        (_float_value(source.get("score")) for source in sources),
                        default=0.0,
                    ),
                    "match_count": len(sources),
                    "model": (
                        settings.azure_chat_deployment
                        if settings.rag_mode == "AZURE_CLOUD"
                        else settings.local_chat_model
                    ),
                    "cache": "HIT" if cache_hit else "MISS",
                },
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
