"""Tests for entity name normalization and resolution."""

from app.ingest.resolution import Resolver, entity_key, normalize
from app.ingest.schema import EntityType


def test_normalize_strips_accents_case_and_apostrophes():
    """Accents, case, curly apostrophes and parentheses are ignored."""
    assert normalize("L’Élysée") == "l'elysee"
    assert normalize("Uber (entreprise)") == "uber"


def test_affair_keys_ignore_the_affaire_prefix():
    """'Affaire Bygmalion', 'l'affaire Bygmalion' and 'Bygmalion' are one affair."""
    keys = {
        entity_key(name, EntityType.AFFAIR)
        for name in ("Affaire Bygmalion", "l'affaire Bygmalion", "Bygmalion")
    }
    assert keys == {"Affair:bygmalion"}


def test_same_name_different_type_gives_different_keys():
    """The Bygmalion company and the Bygmalion affair stay distinct nodes."""
    assert entity_key("Bygmalion", EntityType.COMPANY) != entity_key(
        "Bygmalion", EntityType.AFFAIR
    )


def test_lone_surname_resolves_to_the_only_matching_person():
    """'Sarkozy' maps to Nicolas Sarkozy when he is the only Sarkozy known."""
    resolver = Resolver({"Nicolas Sarkozy": EntityType.PERSON})
    assert resolver.resolve("Sarkozy", EntityType.PERSON) == "Person:nicolas sarkozy"


def test_ambiguous_surname_is_not_merged():
    """'Balkany' stays separate when both Patrick and Isabelle Balkany are known."""
    resolver = Resolver(
        {"Patrick Balkany": EntityType.PERSON, "Isabelle Balkany": EntityType.PERSON}
    )
    assert resolver.resolve("Balkany", EntityType.PERSON) == "Person:balkany"


def test_display_name_prefers_canonical_spelling():
    """The canonical name is shown, not the LLM's spelling."""
    resolver = Resolver({"François Fillon": EntityType.PERSON})
    key = resolver.resolve("francois fillon", EntityType.PERSON)
    assert resolver.display_name(key) == "François Fillon"
