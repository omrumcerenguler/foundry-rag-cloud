"""Command-line interface for ingestion, querying, chat, and evaluation."""

import argparse
import json
import sys
from pathlib import Path
from typing import cast

from config import Settings
from core.models import RAGQueryRequest
from evaluate import evaluate_mode
from factory import get_rag_service
from ingestion import ingest_directory

BASE_DIR = Path(__file__).parent.resolve()


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI parser."""
    parser = argparse.ArgumentParser(description="Enterprise Hybrid RAG")
    parser.add_argument("--ingest", action="store_true", dest="ingest_flag")
    parser.add_argument("--query", dest="query_flag")
    parser.add_argument("--health", action="store_true", dest="health_flag")
    subparsers = parser.add_subparsers(dest="command", required=False)
    subparsers.add_parser("ingest")
    query = subparsers.add_parser("query")
    query.add_argument("question")
    subparsers.add_parser("health")
    subparsers.add_parser("chat")
    evaluation = subparsers.add_parser("eval")
    evaluation.add_argument("--output", default="evaluation.json")
    return parser


def _resolve_command(args: argparse.Namespace) -> str:
    """Resolve positional subcommands and their flag aliases."""
    if args.command:
        return cast(str, args.command)
    if args.ingest_flag:
        return "ingest"
    if args.query_flag is not None:
        args.question = args.query_flag
        return "query"
    if args.health_flag:
        return "health"
    raise ValueError("a command is required")


def main() -> int:
    """Execute the selected CLI command."""
    args = build_parser().parse_args()
    try:
        settings = Settings.from_env()
        command = _resolve_command(args)
        if command == "ingest":
            service = get_rag_service(settings)
            count = ingest_directory(
                BASE_DIR / "data", service.embedding_provider, service.vector_store
            )
            print(f"Ingested {count} chunks.")
        elif command == "query":
            service = get_rag_service(settings)
            response = service.query(RAGQueryRequest(query=args.question))
            print(response.answer)
            print(f"Sources: {', '.join(response.citations) or 'none'}")
            print(f"Latency: {response.latency_seconds:.3f}s")
        elif command == "health":
            service = get_rag_service(settings)
            if service.vector_store.check_health():
                print("Healthy")
                return 0
            print("Unhealthy", file=sys.stderr)
            return 1
        elif command == "chat":
            service = get_rag_service(settings)
            print("Enter a question, or 'exit' to quit.")
            while True:
                question = input("> ").strip()
                if question.lower() in {"exit", "quit"}:
                    break
                if question:
                    response = service.query(RAGQueryRequest(query=question))
                    print(
                        f"{response.answer}\nSources: {', '.join(response.citations) or 'none'}\nLatency: {response.latency_seconds:.3f}s"
                    )
        else:
            summaries = [
                evaluate_mode(mode, settings) for mode in ("LOCAL", "AZURE_CLOUD")
            ]
            output_path = Path(args.output)
            if not output_path.is_absolute():
                output_path = BASE_DIR / output_path
            output_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        return 0
    except (KeyboardInterrupt, EOFError):
        print("Operation cancelled.")
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Operation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
