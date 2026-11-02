"""Retrieval tools over the graph. Every query is a fixed, parameterised template:
the LLM chooses which tool to call and with which names, never the Cypher itself.
"""

import re

from neo4j import Driver

from app.ingest.schema import RelationType

LUCENE_SPECIAL = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/])')
STOPWORDS = {
    "qui", "que", "quoi", "quel", "quelle", "quels", "quelles", "est", "sont", "les", "des",
    "une", "dans", "pour", "avec", "par", "sur", "aux", "the", "and", "what", "who", "which",
    "how", "comment", "pourquoi", "été", "ont", "il", "elle", "ils", "lui", "leur",
}
RRF_K = 60  # reciprocal rank fusion constant
TYPED_RELATIONS = "|".join(relation.value for relation in RelationType)


def lucene_query(text: str) -> str:
    """Turn free text into a safe Lucene OR query of its meaningful words."""
    words = [w for w in re.findall(r"\w+", text.lower()) if len(w) > 2 and w not in STOPWORDS]
    return " OR ".join(LUCENE_SPECIAL.sub(r"\\\1", word) for word in words)


def fuse_rankings(*rankings: list[str], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: merge ranked id lists into one ranking."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def search_chunks(
    driver: Driver, question: str, embedding: list[float], limit: int = 6
) -> list[dict]:
    """Hybrid passage search: vector and full-text rankings fused with RRF."""
    vector = driver.execute_query(
        """
        CALL db.index.vector.queryNodes('chunk_embedding', $k, $embedding)
        YIELD node, score RETURN node.id AS id, score
        """,
        k=limit * 3,
        embedding=embedding,
    ).records
    fulltext = []
    query = lucene_query(question)
    if query:
        fulltext = driver.execute_query(
            """
            CALL db.index.fulltext.queryNodes('chunk_text', $query, {limit: $k})
            YIELD node, score RETURN node.id AS id, score
            """,
            query=query,
            k=limit * 3,
        ).records
    similarity = {record["id"]: record["score"] for record in vector}
    fused = fuse_rankings([r["id"] for r in vector], [r["id"] for r in fulltext])[:limit]
    if not fused:
        return []
    rows = driver.execute_query(
        """
        UNWIND $ids AS id
        MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk {id: id})
        RETURN c.id AS id, c.text AS text, c.section AS section,
               d.title AS document, d.url AS url
        """,
        ids=[chunk_id for chunk_id, _ in fused],
    ).records
    by_id = {row["id"]: dict(row) for row in rows}
    return [
        by_id[chunk_id]
        | {"rrf": round(score, 4), "similarity": round(similarity.get(chunk_id, 0.0), 3)}
        for chunk_id, score in fused
        if chunk_id in by_id
    ]


def find_entities(driver: Driver, names: list[str], per_name: int = 2) -> list[dict]:
    """Match names to entity nodes with the full-text index (tolerates spelling variants)."""
    found: dict[str, dict] = {}
    for name in names:
        query = lucene_query(name)
        if not query:
            continue
        records = driver.execute_query(
            """
            CALL db.index.fulltext.queryNodes('entity_names', $query, {limit: $k})
            YIELD node, score
            WITH node, score, COUNT { (node)--() } AS degree
            RETURN node.key AS key, node.name AS name,
                   [label IN labels(node) WHERE label <> 'Entity'][0] AS type,
                   score, degree
            ORDER BY score DESC, degree DESC
            """,
            query=query,
            k=per_name,
        ).records
        for record in records:
            found.setdefault(record["key"], dict(record) | {"matched": name})
    return list(found.values())


def entity_relations(driver: Driver, key: str, limit: int = 25) -> list[dict]:
    """Typed relations of one entity, strongest first, with provenance."""
    records = driver.execute_query(
        f"""
        MATCH (e:Entity {{key: $key}})-[r:{TYPED_RELATIONS}]-(other:Entity)
        RETURN startNode(r).name AS source, type(r) AS type, endNode(r).name AS target,
               r.stage AS stage, r.final AS final, r.date AS date,
               r.confidence AS confidence, r.support AS support,
               r.evidence AS evidence, r.chunk_ids AS chunk_ids
        ORDER BY r.support DESC, r.confidence DESC
        LIMIT $limit
        """,
        key=key,
        limit=limit,
    ).records
    return [dict(record) for record in records]


def connection_paths(
    driver: Driver, key_a: str, key_b: str, max_hops: int = 4, limit: int = 3
) -> list[list[dict]]:
    """Shortest paths between two entities over typed relations only (bounded length)."""
    records = driver.execute_query(
        f"""
        MATCH (a:Entity {{key: $a}}), (b:Entity {{key: $b}})
        MATCH p = allShortestPaths((a)-[:{TYPED_RELATIONS}*..{int(max_hops)}]-(b))
        RETURN [r IN relationships(p) | {{
            source: startNode(r).name, type: type(r), target: endNode(r).name,
            stage: r.stage, final: r.final, date: r.date,
            evidence: r.evidence, chunk_ids: r.chunk_ids
        }}] AS hops
        LIMIT $limit
        """,
        a=key_a,
        b=key_b,
        limit=limit,
    ).records
    return [record["hops"] for record in records]


def describe_relation(relation: dict) -> str:
    """Human-readable form of a relation, e.g. 'A -[CONVICTED_IN appeal, final, 2021]-> B'."""
    details = [relation.get("stage")] if relation.get("stage") not in (None, "unknown") else []
    if relation.get("final"):
        details.append("final")
    if relation.get("date"):
        details.append(str(relation["date"]))
    label = relation["type"] + (f" {', '.join(details)}" if details else "")
    return f"{relation['source']} -[{label}]-> {relation['target']}"
