"""Topics with Graph Data Science: Louvain communities over the entity graph.

The projected graph links two entities when they are mentioned in the same passage
(weight = number of shared passages) or when a typed relation connects them
(weight TYPED_WEIGHT each). Louvain then groups entities into communities, which
become :Topic nodes; PageRank ranks entities inside each topic.
"""

from neo4j import Driver

from app.agent.retrieval import TYPED_RELATIONS

GRAPH_NAME = "entity_cooccurrence"
TYPED_WEIGHT = 3.0
MIN_TOPIC_SIZE = 3
DROP_GRAPH = "CALL gds.graph.drop($name, false) YIELD graphName RETURN graphName"

PROJECT = f"""
CALL () {{
    MATCH (a:Entity)<-[:MENTIONS]-(:Chunk)-[:MENTIONS]->(b:Entity)
    WHERE elementId(a) < elementId(b)
    RETURN a, b, 1.0 AS w
    UNION ALL
    MATCH (a:Entity)-[:{TYPED_RELATIONS}]-(b:Entity)
    WHERE elementId(a) < elementId(b)
    RETURN a, b, {TYPED_WEIGHT} AS w
}}
WITH a, b, sum(w) AS weight
WITH gds.graph.project($name, a, b, {{relationshipProperties: {{weight: weight}}}},
                       {{undirectedRelationshipTypes: ['*']}}) AS g
RETURN g.nodeCount AS nodes, g.relationshipCount AS relationships
"""


def topic_name(members: list[dict]) -> str:
    """Name a topic after its most central affair, else its two most central entities."""
    ranked = sorted(members, key=lambda m: m["pagerank"], reverse=True)
    affairs = [m["name"] for m in ranked if m["type"] == "Affair"]
    if affairs:
        return affairs[0]
    return " / ".join(m["name"] for m in ranked[:2])


def build_topics(driver: Driver) -> dict:
    """Run Louvain and PageRank, then (re)create the Topic nodes; return a summary."""
    driver.execute_query(DROP_GRAPH, name=GRAPH_NAME)
    projection = driver.execute_query(PROJECT, name=GRAPH_NAME).records[0]
    louvain = driver.execute_query(
        "CALL gds.louvain.write($name, {writeProperty: 'community', "
        "relationshipWeightProperty: 'weight'}) "
        "YIELD communityCount, modularity RETURN communityCount, modularity",
        name=GRAPH_NAME,
    ).records[0]
    driver.execute_query(
        "CALL gds.pageRank.write($name, {writeProperty: 'pagerank', "
        "relationshipWeightProperty: 'weight'}) YIELD nodePropertiesWritten RETURN 1",
        name=GRAPH_NAME,
    )
    driver.execute_query(DROP_GRAPH, name=GRAPH_NAME)

    members = driver.execute_query(
        """
        MATCH (e:Entity) WHERE e.community IS NOT NULL
        RETURN e.community AS community, e.key AS key, e.name AS name,
               [l IN labels(e) WHERE l <> 'Entity'][0] AS type, e.pagerank AS pagerank
        """
    ).records
    communities: dict[int, list[dict]] = {}
    for member in members:
        communities.setdefault(member["community"], []).append(dict(member))
    topics = [
        {"id": community, "name": topic_name(group), "keys": [m["key"] for m in group]}
        for community, group in communities.items()
        if len(group) >= MIN_TOPIC_SIZE
    ]

    driver.execute_query("MATCH (t:Topic) DETACH DELETE t")
    driver.execute_query(
        """
        UNWIND $topics AS topic
        CREATE (t:Topic {id: topic.id, name: topic.name, size: size(topic.keys)})
        WITH t, topic
        UNWIND topic.keys AS key
        MATCH (e:Entity {key: key})
        MERGE (e)-[:IN_TOPIC]->(t)
        """,
        topics=topics,
    )
    return {
        "projected_nodes": projection["nodes"],
        "projected_relationships": projection["relationships"],
        "communities": louvain["communityCount"],
        "modularity": round(louvain["modularity"], 3),
        "topics": len(topics),
    }
