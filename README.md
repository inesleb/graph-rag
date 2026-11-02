# GraphRAG Affaires

Question answering over a knowledge graph built from French Wikipedia articles
on recent French political-financial affairs (Neo4j, FastAPI, LangGraph, Mistral).

## Quick start

```bash
cp .env.example .env      # then set NEO4J_PASSWORD
make install
make corpus               # downloads the articles into data/raw
make docker-run           # Neo4j on :7474, API on :8000/docs
```

Run `make help` for every command.

## Corpus

56 French Wikipedia articles, hand-picked in `data/articles.csv` and downloaded by
`scripts/download_corpus.py` (`make corpus`):

- **20 affairs** with judicial or political developments since 2020 (McKinsey,
  Fonds Marianne, Bygmalion, Fillon, Sarkozy-Kadhafi, assistants FN and MoDem,
  Benalla, Uber Files...). Full text, without the reference sections, capped at
  40,000 characters.
- **36 context pages** for the people, parties, courts, newspapers and companies
  involved. Intro only, so long biographies do not swamp the affairs.

Wikipedia links between these pages are kept (509 links) and become graph edges
without any LLM. Text is licensed CC BY-SA 4.0 by Wikipedia contributors;
`data/corpus_index.json` lists the URL and revision of every article used.

## To write

- Graph schema and diagram
- Architecture
- API examples and screenshots
- Modelling choices and rejected alternatives
- Appendix: comparison with an AI-generated schema
