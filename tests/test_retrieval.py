"""Tests for the retrieval tools (Neo4j replaced by canned records)."""

from app.agent import retrieval
from tests.fakes import FakeDriver


def test_lucene_query_drops_stopwords_and_escapes():
    """Question words are dropped and Lucene operators escaped."""
    assert retrieval.lucene_query("Qui est Nicolas Sarkozy ?") == "nicolas OR sarkozy"
    assert retrieval.lucene_query("c++ (test)") == "test"
    assert retrieval.lucene_query("qui est") == ""


def test_fuse_rankings_rewards_items_in_both_lists():
    """An item ranked in both lists beats items found by only one."""
    fused = retrieval.fuse_rankings(["a", "b"], ["c", "b"])
    assert fused[0][0] == "b"
    assert {item for item, _ in fused} == {"a", "b", "c"}


def test_describe_relation_shows_stage_final_and_date():
    """Unknown stages are hidden; final and date are shown."""
    rel = {"source": "A", "type": "CONVICTED_IN", "target": "B", "stage": "appeal",
           "final": True, "date": "2023"}
    assert retrieval.describe_relation(rel) == "A -[CONVICTED_IN appeal, final, 2023]-> B"
    rel = {"source": "A", "type": "MEMBER_OF", "target": "C", "stage": None}
    assert retrieval.describe_relation(rel) == "A -[MEMBER_OF]-> C"


def test_search_chunks_fuses_vector_and_fulltext():
    """Results come back in fused order with their document and similarity."""
    driver = FakeDriver({
        "db.index.vector": [{"id": "1-0", "score": 0.91}, {"id": "2-0", "score": 0.85}],
        "db.index.fulltext": [{"id": "2-0", "score": 3.0}],
        "UNWIND $ids": lambda params: [
            {"id": i, "text": f"text {i}", "section": "Intro", "document": "Doc", "url": "u"}
            for i in params["ids"]
        ],
    })
    chunks = retrieval.search_chunks(driver, "Bygmalion condamnation", [0.1], limit=2)
    assert [c["id"] for c in chunks] == ["2-0", "1-0"]
    assert chunks[1]["similarity"] == 0.91


def test_find_entities_keeps_the_matched_name():
    """Each entity remembers which name in the question it matched."""
    driver = FakeDriver({"entity_names": [
        {"key": "Person:nicolas sarkozy", "name": "Nicolas Sarkozy", "type": "Person",
         "score": 2.0, "degree": 10},
    ]})
    found = retrieval.find_entities(driver, ["Sarkozy", "?"])
    assert found == [{"key": "Person:nicolas sarkozy", "name": "Nicolas Sarkozy",
                      "type": "Person", "score": 2.0, "degree": 10, "matched": "Sarkozy"}]


def test_relations_and_paths_use_bounded_typed_queries():
    """Path queries only follow typed relations and cap the length."""
    driver = FakeDriver({
        "allShortestPaths": [{"hops": [{"source": "A", "type": "MEMBER_OF", "target": "B"}]}],
        "LIMIT $limit": [{"source": "A", "type": "MEMBER_OF", "target": "B"}],
    })
    assert retrieval.connection_paths(driver, "a", "b", max_hops=3)[0][0]["target"] == "B"
    query = driver.queries[0][0]
    assert "*..3" in query and "MENTIONS" not in query
    assert retrieval.entity_relations(driver, "a")[0]["type"] == "MEMBER_OF"


def test_entity_relations_put_recent_verdicts_first():
    """Legal relations are ranked before the others, so a recent verdict is not cut."""
    driver = FakeDriver({"LIMIT $limit": []})
    retrieval.entity_relations(driver, "a")
    query, params = driver.queries[0]
    assert "CASE WHEN type(r) IN $legal" in query
    assert query.index("$legal") < query.index("r.support DESC")
    assert params["legal"] == ["ACQUITTED_IN", "CONVICTED_IN", "INDICTED_IN"]
