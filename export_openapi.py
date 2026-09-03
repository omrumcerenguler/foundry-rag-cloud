"""Export the FastAPI OpenAPI schema for gateways and client generators."""

import argparse
import json
from pathlib import Path

from api import app


def main() -> int:
    """Write the current application schema to a JSON file."""
    parser = argparse.ArgumentParser(description="Export the RAG API OpenAPI schema")
    parser.add_argument("--output", type=Path, default=Path("openapi.json"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(app.openapi(), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote OpenAPI schema to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
