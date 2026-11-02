"""The GraphRAG agent, as a LangGraph workflow:

    plan -> retrieve -> (enough evidence?) -> answer | refuse

- plan: the LLM extracts the entities in the question and picks tools from an allow-list.
- retrieve: the tools run fixed Cypher templates (see retrieval.py); every call is traced.
- refuse: guardrail, no LLM call, when retrieval found nothing relevant.
- answer: the LLM answers from numbered sources only; citations it invents are dropped.
"""

from typing import Any, Callable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from neo4j import Driver
from pydantic import BaseModel, Field

from app.agent import retrieval

MIN_SIMILARITY = 0.88  # calibrated: off-topic questions score 0.86-0.89, on-topic 0.90-0.95
NOT_IN_CORPUS = (
    "Je ne trouve pas d'information à ce sujet dans le corpus "
    "(articles Wikipédia sur les affaires politico-financières françaises récentes)."
)

PLAN_PROMPT = """You plan how to answer a question over a knowledge graph of recent French
political-financial affairs (people, parties, companies, institutions, media, affairs,
offenses; relations INVOLVED_IN, INDICTED_IN, CONVICTED_IN, ACQUITTED_IN, MEMBER_OF,
WORKED_FOR, REVEALED, INVESTIGATED, CONCERNS).

Question: {question}

Return the entities explicitly named in the question (full names when obvious),
and the tools to use:
- entity_relations: facts about a named entity (who was convicted, roles, parties...)
- connection_paths: how two or more named entities are connected
Passage search always runs; do not list it."""

ANSWER_PROMPT = """Answer the question using ONLY the numbered sources below.
- Passages are [C..], graph relations are [R..] or [P..] (paths).
- Cite the sources you use by their ids. Never cite an id that is not listed.
- Keep "mis en examen", "condamné" (with its stage: first instance, appeal, final) and
  "relaxé" distinct: never present an indictment or a non-final ruling as a final conviction.
- If the sources do not answer the question, set answerable to false.
- Answer in the language of the question, in a few sentences.

Question: {question}

Sources:
{context}"""


class Plan(BaseModel):
    """What the planner decided."""

    entities: list[str] = Field(default_factory=list)
    tools: list[Literal["entity_relations", "connection_paths"]] = Field(default_factory=list)


class Answer(BaseModel):
    """The answer and the sources it relies on."""

    answerable: bool
    answer: str
    citations: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class AgentState(TypedDict, total=False):
    """State passed between the nodes."""

    question: str
    plan: Plan
    chunks: list[dict]
    entities: list[dict]
    relations: list[dict]
    paths: list[list[dict]]
    sources: dict[str, dict]
    tool_calls: list[dict]
    result: dict


def normalize_citation(citation: str) -> str:
    """'[c1]', ' C1 ' and 'C1' all mean source C1."""
    return citation.strip().strip("[]()").strip().upper()


def dedupe_relations(relations: list[dict]) -> list[dict]:
    """The same edge is found once from each of its ends; keep it once."""
    seen, unique = set(), []
    for relation in relations:
        key = (relation["source"], relation["type"], relation["target"], relation.get("stage"))
        if key not in seen:
            seen.add(key)
            unique.append(relation)
    return unique


def _trace(state: AgentState, tool: str, args: dict, count: int) -> None:
    state.setdefault("tool_calls", []).append({"tool": tool, "args": args, "results": count})


def build_sources(state: AgentState) -> dict[str, dict]:
    """Number every retrieved item so the LLM can cite it: C1.., R1.., P1.."""
    sources = {}
    for i, chunk in enumerate(state.get("chunks", []), start=1):
        sources[f"C{i}"] = {"kind": "passage", **chunk}
    for i, relation in enumerate(state.get("relations", []), start=1):
        sources[f"R{i}"] = {
            "kind": "relation", "text": retrieval.describe_relation(relation), **relation
        }
    for i, path in enumerate(state.get("paths", []), start=1):
        text = "  ".join(retrieval.describe_relation(hop) for hop in path)
        sources[f"P{i}"] = {"kind": "path", "text": text, "hops": path}
    return sources


def format_context(sources: dict[str, dict]) -> str:
    """Render the numbered sources for the answer prompt."""
    lines = []
    for source_id, source in sources.items():
        if source["kind"] == "passage":
            lines.append(f"[{source_id}] ({source['document']} - {source['section']}) "
                         f"{source['text']}")
        elif source["kind"] == "relation":
            quote = (source.get("evidence") or [""])[0]
            lines.append(f"[{source_id}] {source['text']}  evidence: \"{quote}\"")
        else:
            lines.append(f"[{source_id}] path: {source['text']}")
    return "\n".join(lines)


def has_evidence(state: AgentState) -> bool:
    """Guardrail: is there anything relevant to answer from?"""
    if state.get("relations") or state.get("paths"):
        return True
    chunks = state.get("chunks", [])
    return bool(chunks) and max(c["similarity"] for c in chunks) >= MIN_SIMILARITY


def build_agent(
    driver: Driver,
    llm: Any,
    embed: Callable[[str], list[float]],
):
    """Compile the agent. `llm` is a LangChain chat model; `embed` embeds a query."""
    planner = llm.with_structured_output(Plan, method="json_schema")
    answerer = llm.with_structured_output(Answer, method="json_schema")

    def plan(state: AgentState) -> AgentState:
        decided = planner.invoke(PLAN_PROMPT.format(question=state["question"])) or Plan()
        return {"plan": decided, "tool_calls": [
            {"tool": "plan", "args": decided.model_dump(), "results": len(decided.entities)}
        ]}

    def retrieve(state: AgentState) -> AgentState:
        question, decided = state["question"], state["plan"]
        chunks = retrieval.search_chunks(driver, question, embed(question))
        _trace(state, "search_passages", {"query": question}, len(chunks))

        entities = retrieval.find_entities(driver, decided.entities) if decided.entities else []
        _trace(state, "find_entities", {"names": decided.entities}, len(entities))
        best = {}  # best-scoring entity for each name in the question
        for entity in entities:
            best.setdefault(entity["matched"], entity)

        relations: list[dict] = []
        if "entity_relations" in decided.tools or not decided.tools:
            for entity in best.values():
                found = retrieval.entity_relations(driver, entity["key"], limit=12)
                _trace(state, "entity_relations", {"entity": entity["name"]}, len(found))
                relations += found

        paths: list[list[dict]] = []
        keys = [entity["key"] for entity in best.values()]
        if "connection_paths" in decided.tools and len(keys) >= 2:
            for a, b in zip(keys, keys[1:]):
                found = retrieval.connection_paths(driver, a, b)
                _trace(state, "connection_paths", {"from": a, "to": b}, len(found))
                paths += found

        return {"chunks": chunks, "entities": list(best.values()),
                "relations": dedupe_relations(relations),
                "paths": paths, "tool_calls": state["tool_calls"]}

    def route(state: AgentState) -> Literal["answer", "refuse"]:
        return "answer" if has_evidence(state) else "refuse"

    def refuse(_state: AgentState) -> AgentState:
        return {"sources": {}, "result": {
            "status": "not_in_corpus", "answer": NOT_IN_CORPUS, "confidence": 0.0,
            "citations": [], "guardrail": "no passage or relation relevant enough",
        }}

    def answer(state: AgentState) -> AgentState:
        sources = build_sources(state)
        prompt = ANSWER_PROMPT.format(question=state["question"], context=format_context(sources))
        reply = answerer.invoke(prompt)
        cited = list(dict.fromkeys(normalize_citation(c) for c in reply.citations))
        valid = [c for c in cited if c in sources]
        invented = sorted(set(cited) - set(valid))
        confidence = reply.confidence
        notes = []
        if invented:
            notes.append(f"dropped citations not in the sources: {invented}")
            confidence *= 0.5
        if reply.answerable and not valid:
            notes.append("answer cites no source")
            confidence *= 0.5
        if not reply.answerable:
            return {"sources": sources, "result": {
                "status": "not_in_corpus", "answer": NOT_IN_CORPUS, "confidence": 0.0,
                "citations": [], "guardrail": "the LLM judged the sources insufficient",
            }}
        return {"sources": sources, "result": {
            "status": "answered", "answer": reply.answer, "confidence": round(confidence, 2),
            "citations": valid, "guardrail": "; ".join(notes) or None,
        }}

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan)
    graph.add_node("retrieve", retrieve)
    graph.add_node("answer", answer)
    graph.add_node("refuse", refuse)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_conditional_edges("retrieve", route)
    graph.add_edge("answer", END)
    graph.add_edge("refuse", END)
    return graph.compile()


def run_agent(agent, question: str) -> dict:
    """Run the agent and shape the API response: answer, sources, paths, trace."""
    state = agent.invoke({"question": question, "tool_calls": []})
    result, sources = state["result"], state.get("sources", {})
    cited = [{"id": c, **_public(sources[c])} for c in result["citations"]]
    return {
        "question": question,
        **result,
        "sources": cited,
        "graph_paths": [s["text"] for s in sources.values() if s["kind"] in ("relation", "path")
                        and s in [sources[c] for c in result["citations"]]],
        "entities": [{"name": e["name"], "type": e["type"]} for e in state.get("entities", [])],
        "tool_calls": state.get("tool_calls", []),
    }


def _public(source: dict) -> dict:
    """The fields of a source worth returning to the client."""
    if source["kind"] == "passage":
        return {"kind": "passage", "document": source["document"], "section": source["section"],
                "url": source["url"], "text": source["text"]}
    if source["kind"] == "relation":
        return {"kind": "relation", "text": source["text"],
                "evidence": source.get("evidence"), "chunk_ids": source.get("chunk_ids")}
    return {"kind": "path", "text": source["text"]}
