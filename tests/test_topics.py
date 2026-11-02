"""Tests for topic naming and the GDS workflow (Neo4j faked)."""

from app.ingest import topics
from tests.fakes import FakeDriver


def test_topic_named_after_most_central_affair():
    """An affair wins over more central people."""
    members = [
        {"name": "Nicolas Sarkozy", "type": "Person", "pagerank": 3.0},
        {"name": "Affaire Bygmalion", "type": "Affair", "pagerank": 2.0},
        {"name": "Affaire Bismuth", "type": "Affair", "pagerank": 1.0},
    ]
    assert topics.topic_name(members) == "Affaire Bygmalion"


def test_topic_without_affair_uses_top_two_entities():
    """Without an affair, the two most central entities name the topic."""
    members = [
        {"name": "A", "type": "Person", "pagerank": 1.0},
        {"name": "B", "type": "Party", "pagerank": 2.0},
        {"name": "C", "type": "Media", "pagerank": 0.5},
    ]
    assert topics.topic_name(members) == "B / A"


def test_build_topics_keeps_only_big_enough_communities():
    """Communities smaller than MIN_TOPIC_SIZE do not become topics."""
    member = {"type": "Person", "pagerank": 1.0}
    driver = FakeDriver({
        "gds.graph.project": [{"nodes": 5, "relationships": 7}],
        "gds.louvain.write": [{"communityCount": 2, "modularity": 0.41234}],
        "e.community AS community": [
            {"community": 1, "key": f"k{i}", "name": f"n{i}", **member} for i in range(3)
        ] + [{"community": 2, "key": "k9", "name": "n9", **member}],
    })
    summary = topics.build_topics(driver)
    assert summary == {"projected_nodes": 5, "projected_relationships": 7,
                       "communities": 2, "modularity": 0.412, "topics": 1}
    created = next(params for query, params in driver.queries if "CREATE (t:Topic" in query)
    assert [t["id"] for t in created["topics"]] == [1]
