"""Tests for splitting articles into passages."""

from app.ingest.chunking import chunk_text, split_sections

ARTICLE = """Intro paragraph.

== Faits ==
First fact.
Second fact.

=== Détail ===
A detail."""


def test_split_sections_keeps_titles_and_intro():
    """Text before the first heading is the introduction; sub-sections are kept."""
    assert split_sections(ARTICLE) == [
        ("Introduction", "Intro paragraph."),
        ("Faits", "First fact.\nSecond fact."),
        ("Détail", "A detail."),
    ]


def test_chunks_never_cross_sections():
    """Each chunk belongs to exactly one section."""
    chunks = chunk_text(ARTICLE, max_chars=1000)
    assert [c["section"] for c in chunks] == ["Introduction", "Faits", "Détail"]


def test_long_sections_are_split_on_paragraphs():
    """A section longer than max_chars is cut between paragraphs."""
    text = "\n".join(["x" * 40] * 5)
    chunks = chunk_text(text, max_chars=100)
    assert all(len(chunk["text"]) <= 100 for chunk in chunks)
    assert sum(chunk["text"].count("x" * 40) for chunk in chunks) == 5
