"""Reproducible retrieval, grounding, and latency evaluation."""

import argparse
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from config import Settings
from core.models import RAGQueryRequest
from factory import get_rag_service
from ingestion import ingest_directory

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.resolve()


def _questions(path: Path) -> list[str]:
    """Read non-empty question lines."""
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _unavailable(mode: str, error: Exception, output: Path | None) -> dict[str, object]:
    """Return and optionally persist a structured disabled-mode result."""
    summary: dict[str, object] = {
        "mode": mode,
        "question_count": 0,
        "error": str(error),
    }
    print(f"{mode} | unavailable | {error}")
    if output:
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_mode(
    mode: str, settings: Settings | None = None, output: Path | None = None
) -> dict[str, object]:
    """Evaluate the bundled question set, preserving the legacy CLI contract."""
    resolved = (settings or Settings.from_env()).model_copy(update={"rag_mode": mode})
    try:
        service = get_rag_service(resolved)
        ingest_directory(
            BASE_DIR / "data", service.embedding_provider, service.vector_store
        )
    except (RuntimeError, ValueError, OSError) as exc:
        return _unavailable(mode, exc, output)
    rows: list[dict[str, object]] = []
    for question in _questions(BASE_DIR / "questions" / "questions.txt"):
        started = perf_counter()
        response = service.query(RAGQueryRequest(query=question, top_k=3))
        latency = perf_counter() - started
        logger.info(
            "Evaluation query latency",
            extra={
                "retrieval_ms": None,
                "generation_ms": None,
                "total_latency_ms": latency * 1000,
            },
        )
        terms = {
            term.lower().strip("?,.!:") for term in question.split() if len(term) > 3
        }
        hit = any(
            any(term in result.text.lower() for term in terms)
            for result in response.sources
        )
        citation_ok = bool(response.citations) and all(
            citation in {result.source_id for result in response.sources}
            for citation in response.citations
        )
        rows.append(
            {
                "question": question,
                "top_k_hit": hit,
                "latency_seconds": latency,
                "citation_correct": citation_ok,
            }
        )
    count = len(rows)
    summary: dict[str, object] = {
        "mode": mode,
        "question_count": count,
        "retrieval_top_k_hit_rate": sum(bool(row["top_k_hit"]) for row in rows) / count
        if count
        else 0.0,
        "average_latency_seconds": sum(
            cast(float, row["latency_seconds"]) for row in rows
        )
        / count
        if count
        else 0.0,
        "citation_correctness": sum(bool(row["citation_correct"]) for row in rows)
        / count
        if count
        else 0.0,
        "questions": rows,
    }
    print("Mode | Top-K hit rate | Avg latency (s) | Citation correctness")
    print(
        f"{mode} | {summary['retrieval_top_k_hit_rate']:.2%} | {summary['average_latency_seconds']:.3f} | {summary['citation_correctness']:.2%}"
    )
    if output:
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _load_cases(dataset: Path) -> list[dict[str, Any]]:
    """Load a JSON list or an object containing a ``cases`` list."""
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list) or not all(isinstance(case, dict) for case in cases):
        raise ValueError("dataset must be a JSON list of case objects")
    return cast(list[dict[str, Any]], cases)


def _expected_sources(case: dict[str, Any]) -> set[str]:
    """Read the supported expected-source field from a benchmark case."""
    values = case.get("relevant_sources", case.get("relevant_source_ids", []))
    if not isinstance(values, list) or not all(
        isinstance(value, str) for value in values
    ):
        raise ValueError("relevant_sources must be a list of strings")
    return set(values)


def evaluate_dataset(
    dataset: Path,
    settings: Settings | None = None,
    output: Path | None = None,
    data_dir: Path | None = None,
) -> dict[str, object]:
    """Evaluate benchmark cases and return reproducible aggregate metrics."""
    cases = _load_cases(dataset)
    resolved = settings or Settings.from_env()
    service = get_rag_service(resolved)
    if data_dir is not None:
        ingest_directory(data_dir, service.embedding_provider, service.vector_store)
    rows: list[dict[str, object]] = []
    for case in cases:
        question = case.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("every case requires a non-empty question")
        expected = _expected_sources(case)
        started = perf_counter()
        response = service.query(RAGQueryRequest(query=question, top_k=3))
        latency = perf_counter() - started
        logger.info(
            "Evaluation query latency",
            extra={
                "retrieval_ms": None,
                "generation_ms": None,
                "total_latency_ms": latency * 1000,
            },
        )
        retrieved = [source.source_id for source in response.sources]
        ranks = [
            rank for rank, source_id in enumerate(retrieved, 1) if source_id in expected
        ]
        precision = sum(source_id in expected for source_id in retrieved[:3]) / 3.0
        reciprocal_rank = 1.0 / ranks[0] if ranks else 0.0
        cited = set(response.citations)
        grounding = bool(cited) and cited.issubset(set(retrieved))
        answer_terms = case.get("answer_terms", [])
        if answer_terms:
            if not isinstance(answer_terms, list) or not all(
                isinstance(term, str) for term in answer_terms
            ):
                raise ValueError("answer_terms must be a list of strings")
            answer = response.answer.lower()
            grounding = grounding and all(
                term.lower() in answer for term in answer_terms
            )
        rows.append(
            {
                "question": question,
                "precision_at_3": precision,
                "reciprocal_rank": reciprocal_rank,
                "citation_grounding": grounding,
                "latency_seconds": latency,
            }
        )
    count = len(rows)
    summary: dict[str, object] = {
        "dataset": str(dataset),
        "mode": resolved.rag_mode,
        "case_count": count,
        "precision_at_3": sum(cast(float, row["precision_at_3"]) for row in rows)
        / count
        if count
        else 0.0,
        "mrr": sum(cast(float, row["reciprocal_rank"]) for row in rows) / count
        if count
        else 0.0,
        "faithfulness_citation_grounding": sum(
            bool(row["citation_grounding"]) for row in rows
        )
        / count
        if count
        else 0.0,
        "average_latency_seconds": sum(
            cast(float, row["latency_seconds"]) for row in rows
        )
        / count
        if count
        else 0.0,
        "cases": rows,
    }
    print(
        f"Dataset | cases={count} | P@3={summary['precision_at_3']:.2%} | MRR={summary['mrr']:.2%} | grounding={summary['faithfulness_citation_grounding']:.2%} | avg_latency={summary['average_latency_seconds']:.3f}s"
    )
    if output:
        output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    """Run the dataset evaluator or the legacy mode evaluator."""
    parser = argparse.ArgumentParser(
        description="Evaluate the Enterprise Hybrid RAG service"
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("eval_report.json"))
    parser.add_argument("--mode", choices=("LOCAL", "AZURE_CLOUD"), default=None)
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "data")
    args = parser.parse_args()
    if args.dataset:
        settings = Settings.from_env()
        if args.mode:
            settings = settings.model_copy(update={"rag_mode": args.mode})
        evaluate_dataset(args.dataset, settings, args.output, args.data_dir)
    else:
        evaluate_mode(args.mode or Settings.from_env().rag_mode, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
