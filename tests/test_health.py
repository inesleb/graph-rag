"""Tests for the health endpoint."""

from fastapi.testclient import TestClient

from app import main

client = TestClient(main.app)


def test_health_reports_neo4j_up(monkeypatch):
    """The endpoint says Neo4j is up when the connection check passes."""
    monkeypatch.setattr(main, "check_connection", lambda: True)
    assert client.get("/health").json() == {"status": "ok", "neo4j": "up"}


def test_health_reports_neo4j_down(monkeypatch):
    """The endpoint still answers when Neo4j is unreachable."""
    monkeypatch.setattr(main, "check_connection", lambda: False)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["neo4j"] == "down"
