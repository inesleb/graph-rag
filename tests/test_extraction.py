"""Tests for the validation of LLM extractions (the hallucination guard)."""

from app.ingest.extraction import build_prompt, quote_in_text, validate
from app.ingest.schema import Extraction

PASSAGE = (
    "Le 1er mars 2021, Nicolas Sarkozy est condamné à trois ans de prison "
    "dont un ferme pour corruption et trafic d’influence."
)


def make(relations, entities=None):
    """Build an Extraction from plain dicts."""
    entities = entities or [
        {"name": "Nicolas Sarkozy", "type": "Person"},
        {"name": "Affaire des écoutes", "type": "Affair"},
        {"name": "Mediapart", "type": "Media"},
    ]
    return Extraction.model_validate({"entities": entities, "relations": relations})


def relation(**overrides):
    """A valid CONVICTED_IN relation, with optional overrides."""
    base = {
        "source": "Nicolas Sarkozy",
        "type": "CONVICTED_IN",
        "target": "Affaire des écoutes",
        "evidence": "Nicolas Sarkozy est condamné à trois ans de prison",
        "confidence": 0.9,
        "stage": "first_instance",
        "final": False,
    }
    return base | overrides


def test_quote_matching_ignores_case_accents_and_apostrophes():
    """Typographic differences do not make a real quote fail."""
    assert quote_in_text("corruption et trafic d'influence", PASSAGE)
    assert not quote_in_text("condamné à cinq ans", PASSAGE)


def test_valid_relation_is_kept():
    """A well-formed, quoted relation passes."""
    kept, rejected = validate(make([relation()]), PASSAGE)
    assert len(kept.relations) == 1 and not rejected


def test_invented_quote_is_rejected():
    """A relation whose evidence is not in the passage is dropped."""
    kept, rejected = validate(make([relation(evidence="Sarkozy a été relaxé en appel")]), PASSAGE)
    assert not kept.relations
    assert rejected == {"evidence_not_in_passage": 1}


def test_wrong_entity_types_are_rejected():
    """A person cannot REVEAL an affair: only media can."""
    kept, rejected = validate(make([relation(type="REVEALED")]), PASSAGE)
    assert not kept.relations and rejected == {"wrong_types": 1}


def test_relation_to_unlisted_entity_is_rejected():
    """Both ends of a relation must be among the extracted entities."""
    kept, rejected = validate(make([relation(target="Affaire Bygmalion")]), PASSAGE)
    assert not kept.relations and rejected == {"unknown_entity": 1}


def test_stage_is_cleared_on_non_legal_relations():
    """Only legal relations carry a judicial stage."""
    entities = [
        {"name": "Nicolas Sarkozy", "type": "Person"},
        {"name": "Les Républicains", "type": "Party"},
    ]
    rel = relation(type="MEMBER_OF", target="Les Républicains", evidence="Nicolas Sarkozy est")
    kept, _ = validate(make([rel], entities), PASSAGE)
    assert kept.relations[0].stage is None and kept.relations[0].final is None


def test_prompt_names_the_affair_and_canonical_entities():
    """The prompt tells the LLM which affair the article is about."""
    prompt = build_prompt(
        {"section": "Procès", "text": PASSAGE},
        {"title": "Affaire des écoutes (Sarkozy)", "kind": "affair"},
        ["Nicolas Sarkozy", "Mediapart"],
    )
    assert 'about the affair "Affaire des écoutes (Sarkozy)"' in prompt
    assert "Nicolas Sarkozy, Mediapart" in prompt and PASSAGE in prompt


def test_legal_relation_may_target_an_offense():
    """'condamné pour corruption' links the person to the offense node."""
    entities = [
        {"name": "Nicolas Sarkozy", "type": "Person"},
        {"name": "corruption", "type": "Offense"},
    ]
    rel = relation(target="corruption")
    kept, rejected = validate(make([rel], entities), PASSAGE)
    assert len(kept.relations) == 1 and not rejected
