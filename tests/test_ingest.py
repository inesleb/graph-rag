"""Tests for the ingestion pipeline (no Neo4j or LLM needed)."""

import httpx
import pytest

from app.ingest import pipeline
from app.ingest.pipeline import assemble, canonical_name, invoke_with_backoff, make_chunks
from app.ingest.schema import EntityType, Extraction

TEXT_1 = "Nicolas Sarkozy est condamné en première instance pour corruption."
TEXT_2 = "En appel, Sarkozy est condamné pour corruption, décision définitive."
DOCUMENTS = [
    {"pageid": 1, "title": "Affaire des écoutes (Sarkozy)", "kind": "affair",
     "entity_type": EntityType.AFFAIR, "text": TEXT_1},
    {"pageid": 2, "title": "Nicolas Sarkozy", "kind": "context",
     "entity_type": EntityType.PERSON, "text": TEXT_2},
]
CHUNKS = [
    {"id": "1-0", "pageid": 1, "position": 0, "section": "Introduction", "text": TEXT_1},
    {"id": "1-1", "pageid": 1, "position": 1, "section": "Appel", "text": TEXT_2},
]


def extraction(source, evidence, stage, final):
    """One CONVICTED_IN extraction."""
    return Extraction.model_validate({
        "entities": [{"name": source, "type": "Person"},
                     {"name": "Affaire des écoutes", "type": "Affair"}],
        "relations": [{"source": source, "type": "CONVICTED_IN",
                       "target": "Affaire des écoutes", "evidence": evidence,
                       "confidence": 0.9, "stage": stage, "final": final, "date": "2021"}],
    })


def test_canonical_name_drops_disambiguation():
    """Only a trailing parenthesis is removed."""
    assert canonical_name("Uber (entreprise)") == "Uber"
    assert canonical_name("Les Républicains") == "Les Républicains"


def test_make_chunks_ids_are_stable():
    """Chunk ids combine the page id and the position."""
    chunks = make_chunks([{"pageid": 7, "text": "Intro.\n== A ==\nTexte."}], 1200)
    assert [chunk["id"] for chunk in chunks] == ["7-0", "7-1"]


def test_assemble_merges_names_and_keeps_one_edge_per_stage():
    """'Sarkozy' and 'Nicolas Sarkozy' are one node; each judicial stage is its own edge."""
    graph = assemble(
        CHUNKS,
        {
            "1-0": extraction(
                "Nicolas Sarkozy", "Nicolas Sarkozy est condamné", "first_instance", False
            ),
            "1-1": extraction(
                "Sarkozy", "Sarkozy est condamné pour corruption, décision définitive", "appeal", True
            ),
        },
        DOCUMENTS,
    )
    keys = {entity["key"] for entity in graph["entities"]}
    assert "Person:nicolas sarkozy" in keys and "Person:sarkozy" not in keys
    stages = {rel["stage"]: rel["props"] for rel in graph["relations"]}
    assert set(stages) == {"first_instance", "appeal"}
    assert stages["appeal"]["final"] is True and stages["first_instance"]["final"] is False
    assert {"pageid": 1, "key": "Affair:ecoutes"} in graph["about"]
    assert graph["report"]["relations_kept"] == 2


def http_error(status):
    """An httpx error with the given status code."""
    request = httpx.Request("POST", "https://api.mistral.ai")
    return httpx.HTTPStatusError("x", request=request, response=httpx.Response(status, request=request))


class FlakyModel:
    """Fails with the given errors, then succeeds."""

    def __init__(self, *errors):
        self.errors = list(errors)

    def invoke(self, prompt):
        """Raise the next error or return a result."""
        if self.errors:
            raise self.errors.pop(0)
        return f"ok: {prompt}"


def test_backoff_retries_rate_limits_server_errors_and_timeouts(monkeypatch):
    """429, 503 and timeouts are retried until the call succeeds."""
    monkeypatch.setattr(pipeline.time, "sleep", lambda seconds: None)
    model = FlakyModel(http_error(429), http_error(503), httpx.ReadTimeout("slow"))
    assert invoke_with_backoff(model, "p") == "ok: p"


def test_backoff_gives_up_with_none(monkeypatch):
    """After the last attempt, None is returned instead of crashing the run."""
    monkeypatch.setattr(pipeline.time, "sleep", lambda seconds: None)
    model = FlakyModel(*[http_error(503)] * pipeline.RATE_LIMIT_RETRIES)
    assert invoke_with_backoff(model, "p") is None


def test_backoff_does_not_retry_client_errors(monkeypatch):
    """A 400 is a bug, not a transient failure: it is raised at once."""
    monkeypatch.setattr(pipeline.time, "sleep", lambda seconds: None)
    with pytest.raises(httpx.HTTPStatusError):
        invoke_with_backoff(FlakyModel(http_error(400)), "p")


def test_final_needs_cassation_or_an_explicit_quote():
    """The LLM's final flag only counts when the stage or the quote supports it."""
    def rel(evidence, final=True):
        return Extraction.model_validate({
            "entities": [], "relations": [{"source": "a", "type": "CONVICTED_IN", "target": "b",
                                           "evidence": evidence, "confidence": 1, "final": final}],
        }).relations[0]

    assert not pipeline.supported_final(rel("la cour d'appel confirme la culpabilité"), "appeal")
    assert pipeline.supported_final(rel("il est définitivement condamné"), "appeal")
    assert pipeline.supported_final(rel("le pourvoi est rejeté"), "cassation")
    assert not pipeline.supported_final(rel("définitivement condamné", final=False), "appeal")
