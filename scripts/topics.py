"""Detect topics with GDS (Louvain + PageRank) on the current graph.

Usage: python -m scripts.topics
"""

import json

from app.db import get_driver
from app.ingest.topics import build_topics


def main() -> None:
    """Entry point."""
    print(json.dumps(build_topics(get_driver()), indent=2))


if __name__ == "__main__":
    main()
