"""Read-only graph queries behind the exploration endpoints."""

from neo4j import Driver

from app.agent.retrieval import TYPED_RELATIONS, find_entities

MAX_NODES = 150  # cap on graph endpoints so a hub cannot return the whole graph


def search_entities(driver: Driver, text: str, limit: int) -> list[dict]:
    """Entities matching a name."""
    return find_entities(driver, [text], per_name=limit)


def _subgraph(driver: Driver, seed_query: str, depth: int, **params) -> dict:
    """Nodes within `depth` typed hops of the seed entities, and the edges between them.

    Seeds and neighbours are sorted by PageRank before the MAX_NODES cut, so a large
    topic keeps its most central entities instead of an arbitrary subset."""
    records = driver.execute_query(
        f"""
        {seed_query}
        WITH seed ORDER BY coalesce(seed.pagerank, 0) DESC
        WITH collect(DISTINCT seed) AS seeds
        UNWIND seeds AS seed
        OPTIONAL MATCH (seed)-[:{TYPED_RELATIONS}*1..{int(depth)}]-(other:Entity)
        WITH seeds, other ORDER BY coalesce(other.pagerank, 0) DESC
        WITH seeds, collect(DISTINCT other) AS others
        WITH [n IN seeds + others WHERE n IS NOT NULL][..$max_nodes] AS nodes
        UNWIND nodes AS a
        OPTIONAL MATCH (a)-[r:{TYPED_RELATIONS}]->(b) WHERE b IN nodes
        RETURN nodes, collect(DISTINCT r) AS rels
        """,
        max_nodes=MAX_NODES,
        **params,
    ).records
    if not records or not records[0]["nodes"]:
        return {"nodes": [], "edges": []}
    nodes, rels = records[0]["nodes"], records[0]["rels"]
    return {
        "nodes": [
            {"key": n["key"], "name": n["name"],
             "type": next((label for label in n.labels if label != "Entity"), None)}
            for n in nodes
        ],
        "edges": [
            {"source": r.start_node["key"], "type": r.type, "target": r.end_node["key"],
             "stage": r.get("stage"), "final": r.get("final"), "date": r.get("date"),
             "support": r.get("support"), "confidence": r.get("confidence")}
            for r in rels
        ],
    }


def entity_graph(driver: Driver, key: str, depth: int) -> dict:
    """Neighbourhood of one entity."""
    return _subgraph(driver, "MATCH (seed:Entity {key: $key})", depth, key=key)


def topic_graph(driver: Driver, topic_id: int, depth: int) -> dict:
    """Entities of a topic plus their neighbours up to `depth` hops."""
    graph = _subgraph(
        driver, "MATCH (seed:Entity)-[:IN_TOPIC]->(:Topic {id: $topic_id})", depth,
        topic_id=topic_id,
    )
    graph["topic"] = topic_id
    return graph


def list_topics(driver: Driver) -> list[dict]:
    """Topics with their size and most central members."""
    records = driver.execute_query(
        """
        MATCH (t:Topic)<-[:IN_TOPIC]-(e:Entity)
        WITH t, e ORDER BY e.pagerank DESC
        RETURN t.id AS id, t.name AS name, count(e) AS size, collect(e.name)[..5] AS top_entities
        ORDER BY size DESC
        """
    ).records
    return [dict(record) for record in records]


def stats(driver: Driver) -> dict:
    """Counts of nodes per label and relationships per type."""
    nodes = driver.execute_query(
        "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS count ORDER BY count DESC"
    ).records
    rels = driver.execute_query(
        "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY count DESC"
    ).records
    return {
        "nodes": {r["label"]: r["count"] for r in nodes},
        "relationships": {r["type"]: r["count"] for r in rels},
    }
