"""Download the corpus listed in data/articles.csv from French Wikipedia.

Usage: python -m scripts.download_corpus [--limit N]
"""

import argparse
import json
from pathlib import Path

from app.ingest.wikipedia import download, new_session, read_article_list


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--articles", type=Path, default=Path("data/articles.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--max-chars", type=int, default=40_000, help="cap per article")
    parser.add_argument("--limit", type=int, default=None, help="max articles (for testing)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    args = parse_args(argv)
    entries = read_article_list(args.articles)[: args.limit]
    index = download(entries, args.out, new_session(), args.max_chars)
    index_path = args.out.parent / "corpus_index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(entry["chars"] for entry in index)
    print(f"Saved {len(index)} articles ({total:,} characters) to {args.out}")


if __name__ == "__main__":
    main()
