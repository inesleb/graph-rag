"""Exploration queries: bounded and ordered by centrality."""

from app import graph_queries
from tests.fakes import FakeDriver


def test_topic_graph_keeps_central_entities_before_the_cap():
    """Seeds and neighbours are sorted by PageRank before the node cap is applied."""
    driver = FakeDriver({"IN_TOPIC": [{"nodes": [], "rels": []}]})
    graph = graph_queries.topic_graph(driver, 3, depth=1)
    query, params = driver.queries[0]
    assert graph == {"nodes": [], "edges": [], "topic": 3}
    assert query.index("seed.pagerank") < query.index("$max_nodes")
    assert query.index("other.pagerank") < query.index("$max_nodes")
    assert params["max_nodes"] == graph_queries.MAX_NODES
