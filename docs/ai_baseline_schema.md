# AI-generated baseline schema

Generated on 2026-11-02 with `ministral-14b-latest` (temperature 0), used as the baseline for the anti-genericity appendix.

**Prompt**

> Propose a Neo4j graph schema (node labels with properties, relationship types) for a GraphRAG question-answering assistant built from about 50 French Wikipedia articles about recent French political-financial affairs (corruption, illegal campaign financing, embezzlement...). Be concise: list the nodes, the relationships, and the indexes.

**Answer (verbatim)**

Here’s a concise **Neo4j graph schema** for a **GraphRAG QA assistant** on French political-financial corruption, optimized for retrieval-augmented generation (RAG):

---

### **Nodes (Labels + Key Properties)**
1. **Person**
   - `name` (string)
   - `role` (e.g., "politician", "businessman", "judge")
   - `party` (string, e.g., "LR", "LREM")
   - `scandals` (list of `Scandal` IDs)
   - `sentence` (string, if convicted)
   - `year_born` (integer)

2. **Organization**
   - `name` (string)
   - `type` (e.g., "political party", "company", "NGO")
   - `sector` (e.g., "construction", "finance")
   - `scandals` (list of `Scandal` IDs)

3. **Scandal**
   - `name` (string)
   - `year` (integer)
   - `description` (string)
   - `status` (e.g., "ongoing", "closed", "convictions")
   - `investigating_authority` (string, e.g., "Parquet national financier")
   - `legal_basis` (string, e.g., "loi Sapin II")

4. **Law**
   - `name` (string)
   - `year` (integer)
   - `description` (string)
   - `articles` (list of affected legal articles)

5. **Transaction**
   - `amount` (float, in €)
   - `currency` (string)
   - `date` (date)
   - `type` (e.g., "donation", "kickback", "offshore_flow")
   - `source` (string, e.g., "campaign fund")
   - `destination` (string, e.g., "private account")

6. **Document**
   - `title` (string)
   - `source` (string, e.g., "Le Monde", "Parlement français")
   - `date` (date)
   - `url` (string)
   - `content` (string, embeddable text)

7. **Location**
   - `name` (string, e.g., "Paris", "Panama")
   - `type` (e.g., "country", "city", "offshore_paradise")
   - `jurisdiction` (string, e.g., "French", "Swiss")

8. **EntityMention** *(for RAG grounding)*
   - `text` (string, exact phrase from `Document`)
   - `context` (string, surrounding sentence)
   - `document_id` (reference to `Document`)

---

### **Relationships**
1. **PERSON_INVOLVED_IN** → `(Person)-[:PERSON_INVOLVED_IN]->(Scandal)`
   - `role` (e.g., "accused", "witness", "judge")

2. **ORGANIZATION_INVOLVED_IN** → `(Organization)-[:ORGANIZATION_INVOLVED_IN]->(Scandal)`
   - `role` (e.g., "beneficiary", "facilitator")

3. **PERSON_WORKS_AT** → `(Person)-[:PERSON_WORKS_AT]->(Organization)`
   - `start_year` (integer)
   - `end_year` (integer, optional)

4. **SCANDAL_RELATED_TO_LAW** → `(Scandal)-[:SCANDAL_RELATED_TO_LAW]->(Law)`
   - `violation_articles` (list of strings)

5. **PERSON_RECEIVED_TRANSACTION** → `(Person)-[:PERSON_RECEIVED_TRANSACTION]->(Transaction)`
   - `via` (string, e.g., "shell company")

6. **ORGANIZATION_RECEIVED_TRANSACTION** → `(Organization)-[:ORGANIZATION_RECEIVED_TRANSACTION]->(Transaction)`

7. **TRANSACTION_ORIGINATED_IN** → `(Transaction)-[:TRANSACTION_ORIGINATED_IN]->(Location)`

8. **SCANDAL_MENTIONED_IN** → `(Scandal)-[:SCANDAL_MENTIONED_IN]->(Document)`
   - `sentence_count` (integer)

9. **PERSON_MENTIONED_IN** → `(Person)-[:PERSON_MENTIONED_IN]->(Document)`
   - `sentence_count` (integer)

10. **ENTITY_MENTION_IN** → `(EntityMention)-[:ENTITY_MENTION_IN]->(Document)`
    - `start_char` (integer)
    - `end_char` (integer)

11. **LAW_AMENDED_BY** → `(Law)-[:LAW_AMENDED_BY]->(Scandal)` *(bidirectional)*

---

### **Indexes (for Fast Lookups)**
1. **Nodes**:
   - `Person(name)`
   - `Organization(name)`
   - `Scandal(name)`
   - `Law(name)`
   - `Document(title, source)`
   - `Location(name)`

2. **Relationships** *(for traversals)*:
   - `PERSON_INVOLVED_IN` (from `Person`)
   - `SCANDAL_MENTIONED_IN` (from `Scandal`)
   - `ENTITY_MENTION_IN` (from `EntityMention`)

3. **Full-Text Search** *(for RAG)*:
   - Index `Document(content)` (for semantic search with `db.index.fulltext.queryNodes()`).

---

### **Key Traversal Patterns for RAG**
1. **Scandal Context**:
   ```cypher
   MATCH (s:Scandal {name: "Affaire Benalla"})-[:SCANDAL_MENTIONED_IN]->(d:Document)
   RETURN d.content LIMIT 5
   ```

2. **Person’s Network**:
   ```cypher
   MATCH (p:Person {name: "François Fillon"})-[r]->(x)
   RETURN x, type(r) LIMIT 20
   ```

3. **Legal Violations**:
   ```cypher
   MATCH (s:Scandal)-[:SCANDAL_RELATED_TO_LAW]->(l:Law)
   RETURN s.name, l.name, l.articles
   ```

4. **Transaction Flows**:
   ```cypher
   MATCH (t:Transaction)-[:TRANSACTION_ORIGINATED_IN]->(loc:Location {type: "offshore_paradise"})
   RETURN t, loc
   ```

---
**Optimization Notes**:
- Use **vector indexes** (e.g., `db.index.vector.createNodeIndex`) for `Document` embeddings if using hybrid search.
- **Periodic commits** to avoid memory issues with 50+ articles.
- **Relationship properties** (e.g., `role`, `amount`) enable precise filtering.
