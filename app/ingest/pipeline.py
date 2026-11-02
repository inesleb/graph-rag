"""Ingestion pipeline: build the knowledge graph from the downloaded corpus.

Steps: chunk the articles, embed the chunks, extract entities and relations with
the LLM, validate them, resolve entity names, then write everything to Neo4j.

LLM extractions are cached in data/extracted/ (one file per chunk, before
validation), so a rerun only calls the LLM for new chunks and the graph can be
rebuilt without an API key. Embeddings are cached in data/cache/.
"""

import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from app.config import get_settings
from app.db import get_driver
from app.ingest import graph_writer
from app.ingest.chunking import chunk_text
from app.ingest.extraction import build_prompt, validate
from app.ingest.resolution import PARENTHESIS, Resolver, normalize
from app.ingest.schema import LEGAL_RELATIONS, EntityType, Extraction, RelationType
from app.llm import get_chat_model, get_embeddings

MAX_EVIDENCE = 3  # quotes kept per relation
FINAL_WORDS = ("definiti",)  # "définitif", "définitivement", accents removed by normalize()


def supported_final(relation, stage: str | None) -> bool:
    """A decision counts as final only if the cassation stage or the quote itself says so.

    The LLM's `final` flag alone is not trusted: it marked appeal rulings as final
    while a cassation appeal was pending.
    """
    if not relation.final:
        return False
    quote = normalize(relation.evidence)
    return stage == "cassation" or any(word in quote for word in FINAL_WORDS)
RATE_LIMIT_RETRIES = 6


def _retryable(error: Exception) -> bool:
    """Rate limits, server errors and timeouts are worth retrying; other errors are not."""
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code == 429 or error.response.status_code >= 500
    return isinstance(error, httpx.TransportError)


def invoke_with_backoff(model, prompt: str):
    """Call the model, waiting longer after each retryable failure; None if all attempts fail.

    LangChain's own retries do not cover 429 and 503 responses from Mistral.
    """
    for attempt in range(RATE_LIMIT_RETRIES):
        try:
            return model.invoke(prompt)
        except (httpx.HTTPStatusError, httpx.TransportError) as error:
            if not _retryable(error):
                raise
            time.sleep(2 ** (attempt + 1))
    return None


def load_documents(raw_dir: Path, articles_csv: Path) -> list[dict]:
    """Load the downloaded articles, with the entity type each one is about."""
    with articles_csv.open(encoding="utf-8") as file:
        types = {row["title"]: row["entity_type"] for row in csv.DictReader(file)}
    documents = []
    for path in sorted(raw_dir.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        document["entity_type"] = EntityType(types[document["title"]])
        documents.append(document)
    return documents


def make_chunks(documents: list[dict], max_chars: int) -> list[dict]:
    """Chunk every document; chunk ids are '<pageid>-<position>'."""
    chunks = []
    for document in documents:
        for position, chunk in enumerate(chunk_text(document["text"], max_chars)):
            chunks.append(
                {
                    "id": f"{document['pageid']}-{position}",
                    "pageid": document["pageid"],
                    "position": position,
                    **chunk,
                }
            )
    return chunks


def canonical_name(title: str) -> str:
    """Article title without its disambiguation, e.g. 'Uber (entreprise)' -> 'Uber'."""
    return PARENTHESIS.sub(" ", title).strip() if title.endswith(")") else title


def embed_chunks(chunks: list[dict], cache_path: Path) -> None:
    """Add an 'embedding' to every chunk, calling the API only for uncached ones."""
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    missing = [chunk for chunk in chunks if chunk["id"] not in cache]
    if missing:
        print(f"Embedding {len(missing)} chunks...")
        vectors = get_embeddings().embed_documents([chunk["text"] for chunk in missing])
        cache.update({chunk["id"]: vector for chunk, vector in zip(missing, vectors)})
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache))
    for chunk in chunks:
        chunk["embedding"] = cache[chunk["id"]]


def cached_extractions(chunks: list[dict], extracted: Path) -> dict[str, Extraction]:
    """Return the extractions already in the cache, without calling the LLM."""
    results = {}
    for chunk in chunks:
        path = extracted / f"{chunk['id']}.json"
        if path.exists():
            results[chunk["id"]] = Extraction.model_validate_json(path.read_text(encoding="utf-8"))
    return results


def extract_chunks(
    chunks: list[dict],
    documents: dict[int, dict],
    canonical: list[str],
    extracted: Path,
    workers: int = 3,
) -> dict[str, Extraction]:
    """Return the raw LLM extraction of every chunk, using and filling the cache."""
    extracted.mkdir(parents=True, exist_ok=True)
    model = get_chat_model("extraction").with_structured_output(
        Extraction, method="json_schema"
    )

    def extract(chunk: dict) -> tuple[str, Extraction]:
        path = extracted / f"{chunk['id']}.json"
        if path.exists():
            return chunk["id"], Extraction.model_validate_json(path.read_text(encoding="utf-8"))
        prompt = build_prompt(chunk, documents[chunk["pageid"]], canonical)
        extraction = invoke_with_backoff(model, prompt)
        if extraction is None:  # not cached: the next run retries this chunk
            print(f"  giving up on {chunk['id']} for now")
            return chunk["id"], Extraction()
        path.write_text(extraction.model_dump_json(indent=1), encoding="utf-8")
        return chunk["id"], extraction

    todo = sum(not (extracted / f"{c['id']}.json").exists() for c in chunks)
    print(f"Extracting {todo} chunks with the LLM ({len(chunks) - todo} cached)...")
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (chunk_id, extraction) in enumerate(pool.map(extract, chunks), start=1):
            results[chunk_id] = extraction
            if done % 25 == 0:
                print(f"  {done}/{len(chunks)}")
    return results


def _add_relation(relations: dict, relation, source: str, target: str, chunk_id: str) -> None:
    """Merge one validated relation into its (source, type, target, stage) group."""
    stage = None
    if relation.type in LEGAL_RELATIONS:
        stage = relation.stage.value if relation.stage else "unknown"
    group = relations.setdefault(
        (source, relation.type.value, target, stage),
        {"chunk_ids": [], "evidence": [], "confidence": 0.0, "dates": [], "final": False},
    )
    if chunk_id not in group["chunk_ids"]:
        group["chunk_ids"].append(chunk_id)
    if len(group["evidence"]) < MAX_EVIDENCE:
        group["evidence"].append(relation.evidence)
    group["confidence"] = max(group["confidence"], relation.confidence)
    group["dates"] += [relation.date] if relation.date else []
    group["final"] = group["final"] or supported_final(relation, stage)


def _relation_rows(relations: dict) -> list[dict]:
    """Turn the relation groups into rows for graph_writer.write_relations."""
    rows = []
    for (source, rel_type, target, stage), group in relations.items():
        props = {
            "chunk_ids": group["chunk_ids"],
            "evidence": group["evidence"],
            "confidence": round(group["confidence"], 2),
            "support": len(group["chunk_ids"]),
            "date": min(group["dates"]) if group["dates"] else None,
        }
        if RelationType(rel_type) in LEGAL_RELATIONS:
            props["final"] = group["final"]
        rows.append(
            {"source": source, "type": rel_type, "target": target, "stage": stage, "props": props}
        )
    return rows


def assemble(
    chunks: list[dict], extractions: dict[str, Extraction], documents: list[dict]
) -> dict:
    """Validate, resolve and aggregate the extractions into rows ready for Neo4j."""
    resolver = Resolver({canonical_name(d["title"]): d["entity_type"] for d in documents})
    texts = {chunk["id"]: chunk["text"] for chunk in chunks}
    entities: dict[str, dict] = {}
    relations: dict[tuple, dict] = {}
    report = {"relations_extracted": 0, "relations_rejected": {}}

    about = []
    for document in documents:
        key = resolver.resolve(canonical_name(document["title"]), document["entity_type"])
        entities.setdefault(key, {"type": document["entity_type"].value, "chunk_ids": set()})
        about.append({"pageid": document["pageid"], "key": key})

    for chunk_id, raw in extractions.items():
        report["relations_extracted"] += len(raw.relations)
        extraction, reasons = validate(raw, texts[chunk_id])
        for reason, count in reasons.items():
            rejected = report["relations_rejected"]
            rejected[reason] = rejected.get(reason, 0) + count
        keys = {}
        for entity in extraction.entities:
            keys[entity.name] = resolver.resolve(entity.name, entity.type)
            entities.setdefault(
                keys[entity.name], {"type": entity.type.value, "chunk_ids": set()}
            )["chunk_ids"].add(chunk_id)
        for relation in extraction.relations:
            _add_relation(
                relations, relation, keys[relation.source], keys[relation.target], chunk_id
            )

    return {
        "entities": [
            {
                "key": key,
                "name": resolver.display_name(key),
                "type": row["type"],
                "chunk_ids": sorted(row["chunk_ids"]),
            }
            for key, row in entities.items()
        ],
        "relations": _relation_rows(relations),
        "about": about,
        "report": {
            "documents": len(documents),
            "chunks": len(chunks),
            "entities": len(entities),
            **report,
            "relations_kept": sum(len(g["chunk_ids"]) for g in relations.values()),
            "relation_edges": len(relations),
        },
    }


def run(paths: dict[str, Path], use_llm: bool = True, reset: bool = False,
        chunk_chars: int = 1200, workers: int = 3, limit: int | None = None) -> dict:
    """Run the whole pipeline and write the graph; return the ingestion report.

    `paths` needs: raw, articles, extracted, embeddings.
    """
    documents = load_documents(paths["raw"], paths["articles"])[:limit]
    chunks = make_chunks(documents, chunk_chars)
    print(f"{len(documents)} documents, {len(chunks)} chunks")

    embed_chunks(chunks, paths["embeddings"])
    if use_llm:
        canonical = [canonical_name(d["title"]) for d in documents]
        extractions = extract_chunks(
            chunks, {d["pageid"]: d for d in documents}, canonical, paths["extracted"], workers
        )
    else:
        extractions = cached_extractions(chunks, paths["extracted"])
    graph = assemble(chunks, extractions, documents)

    driver = get_driver()
    if reset:
        graph_writer.clear_graph(driver)
    graph_writer.create_schema(driver, get_settings().embedding_dimensions)
    graph_writer.write_documents(driver, documents)
    graph_writer.write_chunks(driver, chunks)
    graph_writer.write_entities(driver, graph["entities"])
    graph_writer.write_about(driver, graph["about"])
    graph_writer.write_relations(driver, graph["relations"])
    return graph["report"]
