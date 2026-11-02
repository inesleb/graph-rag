"""Write documents, chunks, entities and relations to Neo4j.

Labels and relationship types cannot be Cypher parameters, so they are inserted
from the EntityType / RelationType enums only, never from LLM output directly.
"""

from neo4j import Driver

from app.ingest.schema import LEGAL_RELATIONS, EntityType, RelationType

SCHEMA_QUERIES = [
    "CREATE CONSTRAINT document_pageid IF NOT EXISTS "
    "FOR (d:Document) REQUIRE d.pageid IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE",
    "CREATE INDEX document_title IF NOT EXISTS FOR (d:Document) ON (d.title)",
    "CREATE FULLTEXT INDEX entity_names IF NOT EXISTS FOR (e:Entity) ON EACH [e.name]",
    "CREATE FULLTEXT INDEX chunk_text IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]",
]

VECTOR_INDEX = (
    "CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS FOR (c:Chunk) ON c.embedding "
    "OPTIONS {{indexConfig: {{`vector.dimensions`: {dimensions}, "
    "`vector.similarity_function`: 'cosine'}}}}"
)


def create_schema(driver: Driver, dimensions: int) -> None:
    """Create constraints and indexes (idempotent)."""
    for query in [*SCHEMA_QUERIES, VECTOR_INDEX.format(dimensions=dimensions)]:
        driver.execute_query(query)


def write_documents(driver: Driver, documents: list[dict]) -> None:
    """Create Document nodes and the LINKS_TO edges between them."""
    driver.execute_query(
        """
        UNWIND $docs AS doc
        MERGE (d:Document {pageid: doc.pageid})
        SET d.title = doc.title, d.url = doc.url, d.revid = doc.revid, d.kind = doc.kind
        """,
        docs=[
            {key: doc[key] for key in ("pageid", "title", "url", "revid", "kind")}
            for doc in documents
        ],
    )
    driver.execute_query(
        """
        UNWIND $links AS link
        MATCH (a:Document {title: link.source}), (b:Document {title: link.target})
        MERGE (a)-[:LINKS_TO]->(b)
        """,
        links=[
            {"source": doc["title"], "target": target}
            for doc in documents
            for target in doc["links"]
        ],
    )


def write_chunks(driver: Driver, chunks: list[dict]) -> None:
    """Create Chunk nodes with embeddings, HAS_CHUNK and NEXT edges."""
    driver.execute_query(
        """
        UNWIND $chunks AS chunk
        MATCH (d:Document {pageid: chunk.pageid})
        MERGE (c:Chunk {id: chunk.id})
        SET c.text = chunk.text, c.section = chunk.section, c.position = chunk.position
        WITH c, d, chunk
        CALL db.create.setNodeVectorProperty(c, 'embedding', chunk.embedding)
        MERGE (d)-[:HAS_CHUNK]->(c)
        """,
        chunks=chunks,
    )
    driver.execute_query(
        """
        MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
        WITH d, c ORDER BY c.position
        WITH d, collect(c) AS chunks
        UNWIND range(0, size(chunks) - 2) AS i
        WITH chunks[i] AS a, chunks[i + 1] AS b
        MERGE (a)-[:NEXT]->(b)
        """
    )


def write_entities(driver: Driver, entities: list[dict]) -> None:
    """Create Entity nodes with their type label, and MENTIONS edges from chunks."""
    for entity_type in EntityType:
        rows = [e for e in entities if e["type"] == entity_type.value]
        if not rows:
            continue
        driver.execute_query(
            f"""
            UNWIND $rows AS row
            MERGE (e:Entity {{key: row.key}})
            SET e:{entity_type.value}, e.name = row.name
            WITH e, row
            UNWIND row.chunk_ids AS chunk_id
            MATCH (c:Chunk {{id: chunk_id}})
            MERGE (c)-[:MENTIONS]->(e)
            """,
            rows=rows,
        )


def write_about(driver: Driver, about: list[dict]) -> None:
    """Link each Document to the entity it is about."""
    driver.execute_query(
        """
        UNWIND $about AS row
        MATCH (d:Document {pageid: row.pageid}), (e:Entity {key: row.key})
        MERGE (d)-[:ABOUT]->(e)
        """,
        about=about,
    )


def write_relations(driver: Driver, relations: list[dict]) -> None:
    """Create typed relations; legal ones get one edge per judicial stage."""
    for relation_type in RelationType:
        rows = [r for r in relations if r["type"] == relation_type.value]
        if not rows:
            continue
        merge_key = " {stage: row.stage}" if relation_type in LEGAL_RELATIONS else ""
        driver.execute_query(
            f"""
            UNWIND $rows AS row
            MATCH (s:Entity {{key: row.source}}), (t:Entity {{key: row.target}})
            MERGE (s)-[r:{relation_type.value}{merge_key}]->(t)
            SET r += row.props
            """,
            rows=rows,
        )


def clear_graph(driver: Driver) -> None:
    """Delete every node and relationship, in batches."""
    # CALL ... IN TRANSACTIONS needs an auto-commit transaction, hence session.run.
    with driver.session() as session:
        session.run("MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 5000 ROWS")
