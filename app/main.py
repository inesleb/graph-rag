"""FastAPI entry point."""

import csv
from functools import lru_cache
from pathlib import Path

import secrets

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse

from app import graph_queries
from app.agent.graph import build_agent, run_agent
from app.api_models import AskResponse, ImportRequest
from app.config import get_settings
from app.db import check_connection, get_driver
from app.ingest import pipeline, wikipedia
from app.llm import get_chat_model, get_embeddings

DATA = Path("data")
PATHS = {
    "raw": DATA / "raw",
    "articles": DATA / "articles.csv",
    "extracted": DATA / "extracted",
    "embeddings": DATA / "cache" / "embeddings.json",
}

STATIC = Path(__file__).parent / "static"
EXAMPLE_QUESTIONS = {
    "conviction": {
        "summary": "Judicial stages",
        "value": "Marine Le Pen a-t-elle été condamnée dans l'affaire des assistants "
        "parlementaires du FN ?",
    },
    "revealed": {"summary": "Who revealed an affair", "value": "Qui a révélé l'affaire Fillon ?"},
    "connection": {
        "summary": "Link between two people",
        "value": "Quel lien entre Nicolas Sarkozy et Claude Guéant ?",
    },
    "off_topic": {
        "summary": "Guardrail: outside the corpus",
        "value": "Quelle est la recette de la tarte tatin ?",
    },
}

app = FastAPI(
    title="GraphRAG Affaires",
    version="0.1.0",
    description=(
        "Questions over a knowledge graph of **recent French political-financial affairs**, "
        "built from French Wikipedia (56 articles, 1,466 entities, 1,205 typed relations).\n\n"
        "Start with **Questions → GET /api/ask**: every answer cites its passages and graph "
        "relations, lists the agent's steps, and says *not in the corpus* instead of guessing.\n\n"
        "A user interface is available at [/](/)."
    ),
    openapi_tags=[
        {"name": "Questions", "description": "Ask the GraphRAG agent."},
        {"name": "Exploration", "description": "Browse topics (Louvain) and entities."},
        {"name": "Corpus", "description": "Add Wikipedia articles and update the graph."},
        {"name": "System", "description": "Health and graph size."},
    ],
)


@lru_cache
def get_agent():
    """Build the agent once."""
    return build_agent(get_driver(), get_chat_model("answer"), get_embeddings().embed_query)


def require_admin_key(x_api_key: str | None = Header(default=None)) -> None:
    """Protect endpoints that change the graph or spend LLM credit."""
    expected = get_settings().admin_api_key
    if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """The web interface."""
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/health", tags=["System"])
def health() -> dict[str, str]:
    """Report whether the API and Neo4j are reachable."""
    return {"status": "ok", "neo4j": "up" if check_connection() else "down"}


@app.get("/api/ask", response_model=AskResponse, tags=["Questions"])
def ask(
    q: str = Query(
        ..., min_length=3, max_length=500, description="Question, in French or English",
        openapi_examples=EXAMPLE_QUESTIONS,
    ),
) -> dict:
    """Answer a question with the GraphRAG agent; returns sources, graph paths and tool calls."""
    return run_agent(get_agent(), q)


@app.post("/api/corpus/import", dependencies=[Depends(require_admin_key)], tags=["Corpus"])
def import_corpus(request: ImportRequest) -> dict:
    """Download new Wikipedia articles (optional) and (re)build the graph incrementally."""
    known = {row["title"] for row in wikipedia.read_article_list(PATHS["articles"])}
    new = [title for title in request.titles if title not in known]
    downloaded = []
    if new:
        try:
            index = wikipedia.download(
                [{"title": title, "kind": request.kind} for title in new],
                PATHS["raw"],
                wikipedia.new_session(),
                known_titles=known,
            )
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        with PATHS["articles"].open("a", encoding="utf-8", newline="") as file:
            writer = csv.writer(file, lineterminator="\n")
            for entry in index:
                writer.writerow([entry["title"], request.kind, request.entity_type])
        downloaded = [entry["title"] for entry in index]
    report = pipeline.run(PATHS, use_llm=request.use_llm)
    return {"downloaded": downloaded, "already_in_corpus": sorted(set(request.titles) - set(new)),
            "report": report}


@app.get("/api/entities", tags=["Exploration"])
def search_entities(q: str = Query(..., min_length=2), limit: int = Query(10, le=50)) -> list:
    """Find entities by name (full-text, tolerant to accents and partial names)."""
    return graph_queries.search_entities(get_driver(), q, limit)


@app.get("/api/entities/{key}/graph", tags=["Exploration"])
def entity_graph(key: str, depth: int = Query(1, ge=1, le=2)) -> dict:
    """Neighbourhood of an entity over typed relations (nodes and edges)."""
    graph = graph_queries.entity_graph(get_driver(), key, depth)
    if not graph["nodes"]:
        raise HTTPException(status_code=404, detail=f"No entity with key {key!r}")
    return graph


@app.get("/api/topics", tags=["Exploration"])
def list_topics() -> list:
    """Topics found by community detection (Louvain), largest first."""
    return graph_queries.list_topics(get_driver())


@app.get("/api/topics/{topic_id}/graph", tags=["Exploration"])
def topic_graph(topic_id: int, depth: int = Query(2, ge=1, le=3)) -> dict:
    """Related concepts: a topic's entities and their relations, up to `depth` hops."""
    graph = graph_queries.topic_graph(get_driver(), topic_id, depth)
    if not graph["nodes"]:
        raise HTTPException(status_code=404, detail=f"No topic {topic_id}")
    return graph


@app.get("/api/stats", tags=["System"])
def stats() -> dict:
    """Node and relationship counts."""
    return graph_queries.stats(get_driver())
