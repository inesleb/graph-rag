"""Split articles into passages along their sections."""

import re

SECTION_HEADING = re.compile(r"^(={2,})\s*(.+?)\s*\1\s*$", re.MULTILINE)


def split_sections(text: str) -> list[tuple[str, str]]:
    """Return (section title, body) pairs; the text before any heading is 'Introduction'."""
    sections = []
    title, start = "Introduction", 0
    for match in SECTION_HEADING.finditer(text):
        sections.append((title, text[start : match.start()]))
        title, start = match.group(2), match.end()
    sections.append((title, text[start:]))
    return [(name, body.strip()) for name, body in sections if body.strip()]


def chunk_text(text: str, max_chars: int = 1200) -> list[dict]:
    """Group paragraphs into passages of at most ~max_chars, never across sections."""
    chunks = []
    for section, body in split_sections(text):
        current = ""
        for paragraph in (p.strip() for p in body.split("\n") if p.strip()):
            if current and len(current) + len(paragraph) + 1 > max_chars:
                chunks.append({"section": section, "text": current})
                current = ""
            current = f"{current}\n{paragraph}" if current else paragraph
        if current:
            chunks.append({"section": section, "text": current})
    return chunks
