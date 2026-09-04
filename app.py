"""Streamlit interface for interactive grounded conversations."""

import html
import hashlib
import json
import os
from pathlib import Path
from time import monotonic
from typing import Any, cast

import httpx
import streamlit as st
import streamlit.components.v1 as components

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
        "🏗️ Architecture",
        (
            "How does SQLite store embeddings and support cosine similarity search?",
            "How does the RAG assistant separate ingestion, storage, retrieval, and generation?",
            "Why is SQLite an appropriate vector store for a small local knowledge base?",
        ),
    ),
    (
        "🔎 Retrieval",
        (
            "How do chunking and embeddings prepare passages for semantic retrieval?",
            "How does the ingestion pipeline connect source text to later search?",
            "How does deterministic citation rendering keep answers grounded?",
        ),
    ),
    (
        "⚙️ Ingestion",
        (
            "What steps transform source documents into searchable RAG vectors?",
            "Why is local AI useful when prompts and documents must remain private?",
            "What happens when I click Ingest Documents in the sidebar?",
        ),
    ),
    (
        "💻 Hardware & Optimization",
        (
            "How does a Python virtual environment isolate project dependencies?",
            "Why should Apple Silicon users choose ARM64 Python and compatible wheels?",
            "How does a virtual environment improve reproducibility across machines?",
            "What package compatibility concerns matter for local AI on macOS?",
        ),
    ),
    (
        "🗺️ Roadmap",
        (
            "What are the three phases and week ranges in the project delivery plan?",
            "What deliverables are planned for the Testing & Wrap-up phase?",
            "What does the six-week project delivery plan cover?",
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
    copy_key = f"copy_{hashlib.sha256(answer.encode()).hexdigest()[:10]}"
    encoded_answer = json.dumps(answer, ensure_ascii=True).replace("</", "<\\/")
    component_html = f"""
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                background: transparent;
                overflow: hidden;
            }}
            .rag-copy-button {{
                background: transparent;
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 7px;
                color: #9baac0;
                cursor: pointer;
                font-family: sans-serif;
                font-size: 0.7rem;
                padding: 0.25rem 0.55rem;
                transition: color 160ms ease, border-color 160ms ease;
            }}
            .rag-copy-button:hover {{
                border-color: #67e8f9;
                color: #f3f7ff;
            }}
        </style>
        <button id="{copy_key}" class="rag-copy-button" type="button">Copy answer</button>
        <script>
            const button = document.getElementById({copy_key!r});
            const answer = {encoded_answer};

            async function copyAnswer() {{
                try {{
                    if (!navigator.clipboard || !window.isSecureContext) {{
                        throw new Error("Clipboard API unavailable");
                    }}
                    await navigator.clipboard.writeText(answer);
                }} catch (error) {{
                    const fallback = document.createElement("textarea");
                    fallback.value = answer;
                    fallback.style.position = "fixed";
                    fallback.style.opacity = "0";
                    document.body.appendChild(fallback);
                    fallback.focus();
                    fallback.select();
                    fallback.setSelectionRange(0, fallback.value.length);
                    const copied = document.execCommand("copy");
                    fallback.remove();
                    if (!copied) {{
                        throw error;
                    }}
                }}
                button.textContent = "✓ Copied!";
                window.setTimeout(() => {{ button.textContent = "Copy answer"; }}, 1500);
            }}

            button.addEventListener("click", () => {{
                copyAnswer().catch(() => {{
                    button.textContent = "Unable to copy";
                    window.setTimeout(() => {{ button.textContent = "Copy answer"; }}, 1500);
                }});
            }});
        </script>
    """
    with st.container(key=copy_key):
        components.html(component_html, height=35, scrolling=False)


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
    """Render all grounded sources in one collapsed citation container."""
    with st.expander(
        f"📚 Source Citations & Match Context · {len(sources)} source(s)",
        expanded=False,
    ):
        for source in sources:
            score = max(0.0, min(1.0, _float_value(source.get("score"))))
            source_id = html.escape(str(source.get("source_id", "unknown")))
            st.markdown(
                f'<div class="rag-citation-heading"><strong>{source_id}</strong>'
                f'<span class="rag-citation-score">match {score:.3f}</span></div>',
                unsafe_allow_html=True,
            )
            st.progress(score, text=f"Relevance strength · {score:.0%}")
            st.caption(str(source.get("text", "")))


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
    st.set_page_config(
        page_title="Microsoft Foundry & Local AI Assistant",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
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
        [data-testid="stSidebar"] * {
            color: #e2e8f0;
        }
        [data-testid="stSidebar"] .stButton > button,
        [data-testid="stSidebar"] [data-testid="stLinkButton"] > a,
        [data-testid="stSidebar"] [data-testid="stDownloadButton"] > button {
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            color: #ffffff !important;
        }
        [data-testid="stSidebar"] .stButton > button:hover,
        [data-testid="stSidebar"] [data-testid="stLinkButton"] > a:hover,
        [data-testid="stSidebar"] [data-testid="stDownloadButton"] > button:hover {
            background: #162032 !important;
            border-color: rgba(129, 140, 248, 0.72) !important;
            color: #ffffff !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.05) !important;
            border-color: rgba(255, 255, 255, 0.1) !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary *,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary svg,
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] * {
            color: #e2e8f0 !important;
            fill: currentColor !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpanderDetails"] {
            background: #162032 !important;
        }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
            color: #94a3b8 !important;
        }
        [data-testid="stSidebar"] [data-testid="stSlider"] label,
        [data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stThumbValue"] {
            color: #e2e8f0 !important;
        }
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: #162032 !important;
            color: #e2e8f0 !important;
            -webkit-text-fill-color: #e2e8f0;
        }
        [data-testid="stSidebar"] [data-baseweb="tab-list"] {
            background: rgba(255, 255, 255, 0.05) !important;
        }
        [data-testid="stSidebar"] [data-baseweb="tab"] {
            color: #94a3b8 !important;
        }
        [data-testid="stSidebar"] [aria-selected="true"][data-baseweb="tab"] {
            background: #162032 !important;
            color: #ffffff !important;
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
            margin-bottom: 0.85rem;
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
        .rag-citation-heading {
            align-items: center;
            display: flex;
            gap: 0.55rem;
            justify-content: space-between;
            margin-top: 0.5rem;
        }
        .rag-citation-score {
            background: rgba(103, 232, 249, 0.09);
            border: 1px solid rgba(103, 232, 249, 0.22);
            border-radius: 999px;
            color: #b9eeea;
            font-size: 0.7rem;
            padding: 0.2rem 0.48rem;
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
        .st-key-question-catalog .stButton > button {
            background: rgba(255, 255, 255, 0.025);
            font-size: 0.78rem;
            min-height: 2.35rem;
            padding: 0.35rem 0.55rem;
            text-align: left;
        }
        .st-key-question-catalog .stButton > button:hover {
            background: rgba(103, 232, 249, 0.08);
            transform: none;
        }
        .st-key-guide-content {
            color: #c8d5e8;
            font-size: 0.82rem;
            line-height: 1.5;
            overflow-wrap: anywhere;
        }
        .st-key-guide-content h3 {
            color: var(--rag-ink);
            font-size: 0.95rem;
            margin: 0.6rem 0 0.25rem;
        }
        .st-key-guide-content ul {
            margin: 0.25rem 0 0.75rem;
            padding-left: 1.1rem;
        }
        .st-key-guide-content li {
            margin-bottom: 0.45rem;
        }
        .rag-onboarding-banner {
            background: linear-gradient(135deg, rgba(103, 232, 249, 0.1), rgba(129, 140, 248, 0.1));
            border: 1px solid rgba(103, 232, 249, 0.3);
            border-radius: 9px;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05), 0 10px 28px rgba(0, 0, 0, 0.12);
            color: #dceeff;
            margin: 0.1rem 0 1.2rem;
            padding: 0.85rem 1rem;
        }
        .rag-onboarding-banner strong {
            color: var(--rag-cyan);
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
    except ValueError as error:
        st.error(_friendly_error(f"RAG configuration error: {error}"))
        return
    except RuntimeError as error:
        if api_mode and str(error) == "backend connection failed":
            st.error(
                "RAG API connection failed. For direct local Streamlit mode, leave "
                "RAG_API_URL unset; for API mode, run the backend and use "
                "http://127.0.0.1:8000."
            )
        else:
            st.error(_friendly_error("RAG service is unavailable. Check the backend configuration."))
        return
    except Exception:
        st.error(_friendly_error("RAG service is unavailable. Check the backend configuration."))
        return
    with st.sidebar:
        st.title("Foundry AI Ops & Architecture")
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
            help="This per-session allowance protects the Microsoft Foundry engineering demo's Azure credits while keeping it available for genuine visitors.",
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
        with st.expander("📚 Engineering Knowledge Base", expanded=False):
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
        with st.expander("💡 Question Library / Catalog", expanded=False):
            st.caption("Browse categorized questions about Foundry and local AI systems engineering.")
            with st.container(key="question-catalog"):
                for category, questions in QUESTION_LIBRARY:
                    st.markdown(f"**{category}**")
                    for index, question in enumerate(questions):
                        if st.button(
                            question,
                            key=f"catalog-{category}-{index}",
                            use_container_width=True,
                        ):
                            st.session_state["pending_prompt"] = question
        with st.expander("📖 Foundry Engineering Guide / Rehber", expanded=False):
            guide_language = st.radio(
                "Guide language",
                ("🇹🇷 Türkçe", "🇬🇧 English"),
                horizontal=True,
                label_visibility="collapsed",
                key="guide_lang_choice",
            )
            with st.container(key="guide-content"):
                if guide_language == "🇹🇷 Türkçe":
                    st.markdown(
                        """
                        ### Bu Engineering Knowledge Base Nedir?
                        Bu Microsoft Foundry ve local AI systems engineering bilgi tabanı; offline inference, RAG ingestion, SQLite vector retrieval, Apple Silicon uyumluluğu ve tekrarlanabilir Python teslimi için doğrulanmış teknik belgeleri bir araya getirir. Asistan yanıt üretmeden önce bu belgeleri okur; Azure OpenAI embeddings (`text-embedding-3-small`) ilgili pasajları bulur, `gpt-4.1-mini` ise kaynak gösteren yanıtı hazırlar.

                        ### Neden Hazır Bir Engineering Corpus Kullanılıyor?
                        Üretim RAG sistemlerinde Foundry mimarisi, local deployment ve retrieval tasarımı gibi teknik konular; düzenlenmiş, sürümlenmiş ve doğrulanmış bir engineering knowledge base üzerinden sorgulanır. Bu yaklaşım bilgi kirliliğini, beklenmedik token maliyet artışlarını ve kötü niyetli prompt injection saldırılarını azaltır.

                        **Gelecek Çalışması:** Çok kiracılı ve birbirinden izole çalışma alanlarına dosya yükleme, PDF/Docx için sürükle-bırak ayrıştırma ve RBAC kontrollü dinamik indeksleme sonraki sürümlerde planlanmaktadır.

                        ### 🚀 Nasıl Kullanılır?
                        1. **Adım 1 (Ingest):** Önce sidebar'daki **'Ingest Documents'** düğmesine tıklayın. Ham metin dosyaları işlenir ve vektör embeddings SQLite'a kaydedilir.
                        2. **Adım 2 (Inspect):** Foundry ve local AI kaynaklarını incelemek için **'Engineering Knowledge Base'** expander'ını açın.
                        3. **Adım 3 (Ask & Explore):** Sidebar'daki **'💡 Question Library'** içinden test edilmiş bir soru seçin; başlangıçtaki **'Suggested Questions'** grid'inde bir chip'e tıklayın; ya da alttaki chat input'a kendi sorunuzu yazın.
                        4. **Adım 4 (Analyze):** Oluşan yanıtı inceleyin, **Source Citations** dökümünü doğrulayın ve latency/confidence telemetry değerlerini kontrol edin.

                        ### Dil Desteği
                        Internal engineering knowledge base içindeki teknik kaynaklar İngilizcedir. En yüksek retrieval doğruluğu ve deterministik source citation eşleşmesi için İngilizce, birincil ve önerilen dildir. Türkçe sorular `gpt-4.1-mini` tarafından desteklenir; ancak embeddings ve teknik kaynak metinleri İngilizce olduğu için diller arası retrieval confidence değişebilir. En iyi citation eşleşmesi için İngilizce sormanız önerilir.

                        ### Ekran ve Özellik Rehberi
                        - **Ingest Documents:** Kitapları kütüphane rafına dizip indekslemek gibidir; belgeleri işler ve vektörleri SQLite'a yazar.
                        - **Confidence & Latency:** Yanıtın matematiksel benzerlik güvenini ve sistemin tepki süresini gösterir.
                        - **Source Citations & Chunks:** Bilginin tam olarak hangi kaynaktan geldiğini gösteren şeffaf kanıt kutusudur.
                        - **Session Quota & Cooldown:** Sistemi koruyan güvenlik kalkanlarıdır: 8 soru kotası ve sorular arasında 4 saniye bekleme.
                        - **Question Library:** Soldaki hazır soru kataloğuyla konuları tek tıkla keşfedip sistemi test edebilirsiniz.
                        """,
                        unsafe_allow_html=False,
                    )
                else:
                    st.markdown(
                        """
                        ### What is this engineering knowledge base?
                        This Microsoft Foundry and local AI systems engineering knowledge base brings together verified technical material on offline inference, RAG ingestion, SQLite vector retrieval, Apple Silicon compatibility, and reproducible Python delivery. Before answering, the assistant reads these documents; Azure OpenAI embeddings (`text-embedding-3-small`) find the relevant passages, while `gpt-4.1-mini` produces a cited response.

                        ### Why Use a Pre-indexed Engineering Corpus?
                        Production RAG systems query a curated, versioned, and verified engineering knowledge base for topics such as Foundry architecture, local deployment, and retrieval design. This helps prevent data pollution, unexpected token-cost spikes, and adversarial prompt injections.

                        **Future Work:** Multi-tenant isolated workspace uploads, drag-and-drop PDF/Docx parsing, and RBAC-controlled dynamic indexing are planned for future releases.

                        ### 🚀 How to Use?
                        1. **Step 1 (Ingest):** First, click **'Ingest Documents'** in the sidebar. It processes raw text files and stores their vector embeddings in SQLite.
                        2. **Step 2 (Inspect):** Open the **'Engineering Knowledge Base'** expander to inspect the Foundry and local AI source documents.
                        3. **Step 3 (Ask & Explore):** Pick a pre-tested question from the **'💡 Question Library'** in the sidebar; click any chip in the initial **'Suggested Questions'** grid; or type your own custom question in the chat input at the bottom.
                        4. **Step 4 (Analyze):** Inspect the generated answer, verify the **Source Citations** breakdown, and check the latency/confidence telemetry.

                        ### Language Support
                        The internal engineering knowledge base sources are in English. English is the primary and recommended language for the highest retrieval accuracy and deterministic source citations. Turkish queries are supported by the underlying `gpt-4.1-mini`, but because the embeddings and technical source texts are in English, cross-lingual retrieval confidence may vary. Asking in English provides the strongest citation matching.

                        ### Screen & Feature Tour
                        - **Ingest Documents:** Like putting books on library shelves and cataloging them; it processes documents and writes their vectors to SQLite.
                        - **Confidence & Latency:** Shows mathematical similarity confidence and how quickly the system responded.
                        - **Source Citations & Chunks:** Provides a transparent evidence box showing exactly which source supplied the information.
                        - **Session Quota & Cooldown:** Security shields that provide an 8-question allowance and a 4-second wait between questions.
                        - **Question Library:** The ready-made catalog on the left lets you explore topics and test the system with one click.
                        """,
                        unsafe_allow_html=False,
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
            <h1>Microsoft Foundry &amp; Local AI Assistant</h1>
            <p>Enterprise Grounded RAG for Local AI Architecture, SQLite Vector Retrieval &amp; Hardware Optimization</p>
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
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["answer"])
            if message.get("sources"):
                _render_citation_inspector(
                    cast(list[dict[str, object]], message["sources"])
                )
            if message.get("observability"):
                _render_response_observability(message)

    if not messages:
        st.markdown(
            """
            <div class="rag-onboarding-banner">
                <strong>⚡ Getting Started &amp; How It Works</strong>
                <p>This RAG assistant retrieves factual context from a local SQLite vector store using Azure OpenAI embeddings (<code>text-embedding-3-small</code> &amp; <code>gpt-4.1-mini</code>).</p>
                <ul>
                    <li><strong>First Run / Re-indexing:</strong> If the indexed chunks show 0 or you update files, click <strong>'Ingest Documents'</strong> in the sidebar to build vector embeddings.</li>
                    <li><strong>Ask Questions:</strong> Click any sample question below or type your own query to get grounded answers with deterministic citations.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
