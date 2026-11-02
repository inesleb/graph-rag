"""Tests for the LangGraph agent, with the LLM and the retrieval tools faked."""
# pylint: disable=redefined-outer-name,unused-argument  # pytest fixtures and fake signatures

import pytest

from app.agent import graph as agent_graph
from app.agent.graph import (
    Answer,
    Plan,
    build_agent,
    dedupe_relations,
    normalize_citation,
    run_agent,
)
from tests.fakes import FakeLLM

CHUNK = {"id": "1-3", "text": "Nicolas Sarkozy est condamné en appel.", "section": "Procès",
         "document": "Affaire des écoutes (Sarkozy)", "url": "https://fr.wikipedia.org/x",
         "rrf": 0.03, "similarity": 0.9}
RELATION = {"source": "Nicolas Sarkozy", "type": "CONVICTED_IN",
            "target": "Affaire des écoutes", "stage": "appeal", "final": True, "date": "2023",
            "confidence": 0.9, "support": 2, "evidence": ["condamné en appel"],
            "chunk_ids": ["1-3"]}
ENTITY = {"key": "Person:nicolas sarkozy", "name": "Nicolas Sarkozy", "type": "Person",
          "score": 2.0, "degree": 5, "matched": "Sarkozy"}


@pytest.fixture
def tools(monkeypatch):
    """Replace the Neo4j-backed tools with canned results; return a call log."""
    calls = []

    def record(name, result):
        def tool(*args, **kwargs):
            calls.append(name)
            return result() if callable(result) else result
        return tool

    monkeypatch.setattr(agent_graph.retrieval, "search_chunks", record("search", [dict(CHUNK)]))
    monkeypatch.setattr(agent_graph.retrieval, "find_entities", record("find", [dict(ENTITY)]))
    monkeypatch.setattr(agent_graph.retrieval, "entity_relations",
                        record("relations", [dict(RELATION)]))
    monkeypatch.setattr(agent_graph.retrieval, "connection_paths", record("paths", []))
    return calls


def make_agent(plan, answer):
    """An agent whose planner and answerer return the given objects."""
    llm = FakeLLM(Plan=[plan], Answer=[answer])
    return build_agent(driver=None, llm=llm, embed=lambda text: [0.0]), llm


def test_answer_cites_sources_and_exposes_paths(tools):
    """A grounded answer returns its passage, its graph relation and the tool trace."""
    agent, llm = make_agent(
        Plan(entities=["Sarkozy"], tools=["entity_relations"]),
        Answer(answerable=True, answer="Condamné en appel, définitivement.",
               citations=["C1", "R1"], confidence=0.8),
    )
    result = run_agent(agent, "Sarkozy a-t-il été condamné ?")
    assert result["status"] == "answered" and result["confidence"] == 0.8
    assert [s["id"] for s in result["sources"]] == ["C1", "R1"]
    assert result["graph_paths"] == [
        "Nicolas Sarkozy -[CONVICTED_IN appeal, final, 2023]-> Affaire des écoutes"
    ]
    assert [c["tool"] for c in result["tool_calls"]] == [
        "plan", "search_passages", "find_entities", "entity_relations"
    ]
    assert "[R1] Nicolas Sarkozy -[CONVICTED_IN" in llm.structured["Answer"].prompts[0]
    assert tools == ["search", "find", "relations"]


def test_invented_citations_are_dropped_and_confidence_lowered(tools):
    """A citation that was never retrieved is removed and flagged."""
    agent, _ = make_agent(
        Plan(entities=["Sarkozy"], tools=["entity_relations"]),
        Answer(answerable=True, answer="...", citations=["C1", "C9"], confidence=0.8),
    )
    result = run_agent(agent, "Sarkozy a-t-il été condamné ?")
    assert result["citations"] == ["C1"]
    assert result["confidence"] == 0.4
    assert "C9" in result["guardrail"]


def test_off_topic_question_is_refused_without_calling_the_answerer(tools, monkeypatch):
    """No relation and only weak passages: the guardrail answers 'not in corpus'."""
    weak = dict(CHUNK, similarity=0.5)
    monkeypatch.setattr(agent_graph.retrieval, "search_chunks", lambda *a, **k: [weak])
    agent, llm = make_agent(Plan(), Answer(answerable=True, answer="x", confidence=1))
    result = run_agent(agent, "Quelle est la recette de la tarte tatin ?")
    assert result["status"] == "not_in_corpus" and result["sources"] == []
    assert llm.structured["Answer"].prompts == []


def test_llm_can_declare_sources_insufficient(tools):
    """answerable=False from the LLM also yields 'not in corpus'."""
    agent, _ = make_agent(
        Plan(entities=["Sarkozy"]),
        Answer(answerable=False, answer="", citations=[], confidence=0.1),
    )
    result = run_agent(agent, "Quel est le plat préféré de Sarkozy ?")
    assert result["status"] == "not_in_corpus"
    assert result["guardrail"] == "the LLM judged the sources insufficient"


def test_connection_paths_run_for_two_entities(tools, monkeypatch):
    """With two resolved entities and the connection tool, paths are searched."""
    other = dict(ENTITY, key="Person:claude gueant", name="Claude Guéant", matched="Guéant")
    monkeypatch.setattr(agent_graph.retrieval, "find_entities",
                        lambda *a, **k: [dict(ENTITY), other])
    agent, _ = make_agent(
        Plan(entities=["Sarkozy", "Guéant"], tools=["connection_paths"]),
        Answer(answerable=True, answer="...", citations=["C1"], confidence=0.7),
    )
    result = run_agent(agent, "Quel lien entre Sarkozy et Guéant ?")
    assert "connection_paths" in [c["tool"] for c in result["tool_calls"]]
    assert "paths" in tools


def test_citations_in_brackets_are_accepted(tools):
    """'[C1]' is the same citation as 'C1', and duplicates are merged."""
    agent, _ = make_agent(
        Plan(entities=["Sarkozy"], tools=["entity_relations"]),
        Answer(answerable=True, answer="...", citations=["[C1]", "C1", " r1 "], confidence=0.9),
    )
    result = run_agent(agent, "Sarkozy a-t-il été condamné ?")
    assert result["citations"] == ["C1", "R1"] and result["guardrail"] is None


def test_normalize_and_dedupe_helpers():
    """Helpers used by the answer and retrieve steps."""
    assert normalize_citation("[p2]") == "P2"
    twice = [dict(RELATION), dict(RELATION), dict(RELATION, stage="first_instance")]
    assert len(dedupe_relations(twice)) == 2
