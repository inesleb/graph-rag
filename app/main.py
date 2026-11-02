"""FastAPI entry point."""

from fastapi import FastAPI

from app.db import check_connection

app = FastAPI(title="GraphRAG Affaires", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Report whether the API and Neo4j are reachable."""
    return {"status": "ok", "neo4j": "up" if check_connection() else "down"}
