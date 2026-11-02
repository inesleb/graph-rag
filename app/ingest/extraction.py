"""LLM extraction of entities and relations from one passage, and its validation."""

import re

from app.ingest.resolution import normalize
from app.ingest.schema import (
    LEGAL_RELATIONS,
    RELATION_RULES,
    EntityType,
    Extraction,
    ExtractedRelation,
)

PROMPT = """You extract a knowledge graph from a passage of a French Wikipedia article
about French political-financial affairs.

Article: {title}
Section: {section}
{about}
Known canonical names (reuse them exactly when the passage refers to them):
{canonical}

Rules:
- Entities: people, parties, companies, institutions (courts, prosecutors, ministries,
  parliament), media, affairs, and offenses (e.g. "détournement de fonds publics",
  "financement illégal de campagne électorale", "recel", "trafic d'influence").
- Use full names ("Nicolas Sarkozy", not "Sarkozy" or "l'ancien président").
- Only extract relations the passage states explicitly. Never infer.
- INDICTED_IN = mis en examen / poursuivi / renvoyé devant un tribunal. The target of a
  legal relation is the affair, or the offense ("condamné pour recel" -> the offense).
  CONVICTED_IN = condamné. ACQUITTED_IN = relaxé / acquitté / non-lieu.
  Do not turn an indictment into a conviction.
- For legal relations, set stage (investigation, first_instance, appeal, cassation)
  and final=true only if the passage says the decision is definitive.
- WORKED_FOR covers jobs and offices (minister, adviser, director, assistant).
- CONCERNS links an affair to an offense it is about.
- evidence must be copied word for word from the passage.
- Answer in the passage's language for names; keep it short.

Passage:
\"\"\"{text}\"\"\"
"""

_SPACES = re.compile(r"\s+")


def build_prompt(chunk: dict, document: dict, canonical: list[str]) -> str:
    """Fill the extraction prompt for one passage."""
    about = (
        f'This article is about the affair "{document["title"]}".'
        if document["kind"] == "affair"
        else f'This article is about "{document["title"]}".'
    )
    return PROMPT.format(
        title=document["title"],
        section=chunk["section"],
        about=about,
        canonical=", ".join(canonical),
        text=chunk["text"],
    )


def _flat(text: str) -> str:
    return _SPACES.sub(" ", normalize(text).replace('"', "").replace("«", "").replace("»", ""))


def quote_in_text(quote: str, text: str) -> bool:
    """True when the quote appears in the text, ignoring case, accents and spacing."""
    quote = _flat(quote)
    return len(quote) >= 8 and quote in _flat(text)


def rejection_reason(
    relation: ExtractedRelation, types: dict[str, EntityType], text: str
) -> str | None:
    """Return why a relation must be dropped, or None if it is valid."""
    if relation.source not in types or relation.target not in types:
        return "unknown_entity"
    sources, targets = RELATION_RULES[relation.type]
    if types[relation.source] not in sources or types[relation.target] not in targets:
        return "wrong_types"
    if not quote_in_text(relation.evidence, text):
        return "evidence_not_in_passage"
    return None


def validate(extraction: Extraction, text: str) -> tuple[Extraction, dict[str, int]]:
    """Keep only the relations that pass every check; count the rejections by reason."""
    types = {entity.name: entity.type for entity in extraction.entities}
    kept, rejected = [], {}
    for relation in extraction.relations:
        reason = rejection_reason(relation, types, text)
        if reason:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        if relation.type not in LEGAL_RELATIONS:
            relation = relation.model_copy(update={"stage": None, "final": None})
        kept.append(relation)
    return Extraction(entities=extraction.entities, relations=kept), rejected
