"""Tests for the API endpoints, with the agent, pipeline and graph queries faked."""
# pylint: disable=redefined-outer-name,unused-argument  # pytest fixtures and fake signatures

from fastapi.testclient import TestClient

from app import main

client = TestClient(main.app)

ANSWER = {
    "question": "q?", "status": "answered", "answer": "Oui.", "confidence": 0.8,
    "citations": ["C1"], "sources": [{"id": "C1", "kind": "passage", "text": "t",
                                      "document": "D", "section": "S", "url": "u"}],
    "graph_paths": [], "entities": [], "guardrail": None,
    "tool_calls": [{"tool": "plan", "args": {}, "results": 0}],
}


def test_health_reports_neo4j_up(monkeypatch):
    """The endpoint says Neo4j is up when the connection check passes."""
    monkeypatch.setattr(main, "check_connection", lambda: True)
    assert client.get("/health").json() == {"status": "ok", "neo4j": "up"}


def test_health_reports_neo4j_down(monkeypatch):
    """The endpoint still answers when Neo4j is unreachable."""
    monkeypatch.setattr(main, "check_connection", lambda: False)
    assert client.get("/health").json()["neo4j"] == "down"


def test_ask_returns_the_agent_answer(monkeypatch):
    """/api/ask passes the question to the agent and validates the response shape."""
    monkeypatch.setattr(main, "get_agent", lambda: "agent")
    monkeypatch.setattr(main, "run_agent", lambda agent, q: ANSWER | {"question": q})
    response = client.get("/api/ask", params={"q": "Qui a révélé l'affaire Fillon ?"})
    assert response.status_code == 200
    assert response.json()["question"] == "Qui a révélé l'affaire Fillon ?"


def test_ask_rejects_too_short_questions():
    """Input validation happens before any LLM call."""
    assert client.get("/api/ask", params={"q": "a"}).status_code == 422


def test_import_downloads_only_new_titles(monkeypatch, tmp_path):
    """Known titles are skipped, new ones downloaded and appended to the article list."""
    articles = tmp_path / "articles.csv"
    articles.write_text("title,kind,entity_type\nAffaire A,affair,Affair\n", encoding="utf-8")
    monkeypatch.setitem(main.PATHS, "articles", articles)
    downloaded = []

    def fake_download(entries, out, session, known_titles=None):
        downloaded.extend(e["title"] for e in entries)
        return [{"title": e["title"]} for e in entries]

    monkeypatch.setattr(main.wikipedia, "download", fake_download)
    monkeypatch.setattr(main.pipeline, "run", lambda paths, use_llm: {"chunks": 1})
    response = client.post("/api/corpus/import",
                           json={"titles": ["Affaire A", "Affaire B"], "use_llm": False})
    assert response.json() == {"downloaded": ["Affaire B"], "already_in_corpus": ["Affaire A"],
                               "report": {"chunks": 1}}
    assert downloaded == ["Affaire B"]
    assert articles.read_text(encoding="utf-8").endswith("Affaire B,affair,Affair\n")


def test_import_unknown_title_is_404(monkeypatch, tmp_path):
    """A title Wikipedia does not know gives a clear 404."""
    articles = tmp_path / "articles.csv"
    articles.write_text("title,kind,entity_type\n", encoding="utf-8")
    monkeypatch.setitem(main.PATHS, "articles", articles)

    def fail(*args, **kwargs):
        raise ValueError("No Wikipedia article named 'Zzz'")

    monkeypatch.setattr(main.wikipedia, "download", fail)
    response = client.post("/api/corpus/import", json={"titles": ["Zzz"]})
    assert response.status_code == 404


def test_graph_endpoints_return_404_when_empty(monkeypatch):
    """Unknown topic or entity: 404 rather than an empty 200."""
    monkeypatch.setattr(main, "get_driver", lambda: None)
    empty = {"nodes": [], "edges": []}
    monkeypatch.setattr(main.graph_queries, "topic_graph", lambda d, t, depth: empty)
    monkeypatch.setattr(main.graph_queries, "entity_graph", lambda d, k, depth: empty)
    assert client.get("/api/topics/99/graph").status_code == 404
    assert client.get("/api/entities/Person:x/graph").status_code == 404
    assert client.get("/api/topics/1/graph", params={"depth": 9}).status_code == 422


def test_listing_endpoints(monkeypatch):
    """Topics, entity search and stats pass through the graph queries."""
    monkeypatch.setattr(main, "get_driver", lambda: None)
    monkeypatch.setattr(main.graph_queries, "list_topics", lambda d: [{"id": 1}])
    monkeypatch.setattr(main.graph_queries, "search_entities", lambda d, q, n: [{"name": q}])
    monkeypatch.setattr(main.graph_queries, "stats", lambda d: {"nodes": {}})
    assert client.get("/api/topics").json() == [{"id": 1}]
    assert client.get("/api/entities", params={"q": "Fillon"}).json() == [{"name": "Fillon"}]
    assert client.get("/api/stats").json() == {"nodes": {}}


def test_import_requires_the_admin_key_when_configured(monkeypatch):
    """With ADMIN_API_KEY set, import is refused without the right header."""
    settings = main.get_settings().model_copy(update={"admin_api_key": "secret"})
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main.pipeline, "run", lambda paths, use_llm: {})
    assert client.post("/api/corpus/import", json={}).status_code == 401
    assert client.post("/api/corpus/import", json={},
                       headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/api/corpus/import", json={},
                       headers={"X-API-Key": "secret"}).status_code == 200


def test_home_serves_the_web_interface():
    """/ returns the HTML interface; Swagger groups endpoints by tag."""
    response = client.get("/")
    assert response.status_code == 200 and "GraphRAG" in response.text
    tags = {tag for path in client.get("/openapi.json").json()["paths"].values()
            for operation in path.values() for tag in operation.get("tags", [])}
    assert tags == {"Questions", "Exploration", "Corpus", "System"}
