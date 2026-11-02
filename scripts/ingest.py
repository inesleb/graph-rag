"""Build the knowledge graph from the downloaded corpus (see app/ingest/pipeline.py).

Usage: python -m scripts.ingest [--reset] [--no-llm] [--limit N]
"""

import argparse
import json
from pathlib import Path

from app.ingest.pipeline import run


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--articles", type=Path, default=Path("data/articles.csv"))
    parser.add_argument("--extracted", type=Path, default=Path("data/extracted"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/cache/embeddings.json"))
    parser.add_argument("--report", type=Path, default=Path("data/ingest_report.json"))
    parser.add_argument("--chunk-chars", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=3, help="parallel LLM calls")
    parser.add_argument("--limit", type=int, default=None, help="only the first N documents")
    parser.add_argument("--reset", action="store_true", help="empty the graph first")
    parser.add_argument(
        "--no-llm", action="store_true", help="use cached extractions only, never call the LLM"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    args = parse_args(argv)
    paths = {
        "raw": args.raw,
        "articles": args.articles,
        "extracted": args.extracted,
        "embeddings": args.embeddings,
    }
    report = run(paths, use_llm=not args.no_llm, reset=args.reset,
                 chunk_chars=args.chunk_chars, workers=args.workers, limit=args.limit)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
