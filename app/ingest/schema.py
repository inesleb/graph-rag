"""Graph vocabulary: the entity types and relation types the LLM may extract.

Everything the LLM returns is validated against these enums and against
RELATION_RULES, so a relation type or a source/target pairing outside the
schema never reaches Neo4j.
"""

from enum import Enum

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Second label put on an :Entity node."""

    PERSON = "Person"
    PARTY = "Party"
    COMPANY = "Company"
    INSTITUTION = "Institution"
    MEDIA = "Media"
    AFFAIR = "Affair"
    OFFENSE = "Offense"


class RelationType(str, Enum):
    """Relationship types between entities."""

    INVOLVED_IN = "INVOLVED_IN"
    INDICTED_IN = "INDICTED_IN"
    CONVICTED_IN = "CONVICTED_IN"
    ACQUITTED_IN = "ACQUITTED_IN"
    MEMBER_OF = "MEMBER_OF"
    WORKED_FOR = "WORKED_FOR"
    REVEALED = "REVEALED"
    INVESTIGATED = "INVESTIGATED"
    CONCERNS = "CONCERNS"


class Stage(str, Enum):
    """Stage of the judicial procedure a legal relation refers to."""

    INVESTIGATION = "investigation"
    FIRST_INSTANCE = "first_instance"
    APPEAL = "appeal"
    CASSATION = "cassation"
    UNKNOWN = "unknown"


LEGAL_RELATIONS = {
    RelationType.INDICTED_IN,
    RelationType.CONVICTED_IN,
    RelationType.ACQUITTED_IN,
}

_ACTORS = {EntityType.PERSON, EntityType.COMPANY, EntityType.PARTY, EntityType.INSTITUTION}
_EMPLOYERS = {EntityType.COMPANY, EntityType.INSTITUTION, EntityType.PARTY, EntityType.MEDIA}

# Legal relations may target the affair or the offense itself
# ("convicted in the Bygmalion affair" / "convicted of illegal campaign financing").
_LEGAL_TARGETS = {EntityType.AFFAIR, EntityType.OFFENSE}

# relation type -> (allowed source types, allowed target types)
RELATION_RULES: dict[RelationType, tuple[set[EntityType], set[EntityType]]] = {
    RelationType.INVOLVED_IN: (_ACTORS, {EntityType.AFFAIR}),
    RelationType.INDICTED_IN: (_ACTORS, _LEGAL_TARGETS),
    RelationType.CONVICTED_IN: (_ACTORS, _LEGAL_TARGETS),
    RelationType.ACQUITTED_IN: (_ACTORS, _LEGAL_TARGETS),
    RelationType.MEMBER_OF: ({EntityType.PERSON}, {EntityType.PARTY, EntityType.INSTITUTION}),
    RelationType.WORKED_FOR: ({EntityType.PERSON}, _EMPLOYERS),
    RelationType.REVEALED: ({EntityType.MEDIA}, {EntityType.AFFAIR}),
    RelationType.INVESTIGATED: ({EntityType.INSTITUTION}, {EntityType.AFFAIR}),
    RelationType.CONCERNS: ({EntityType.AFFAIR}, {EntityType.OFFENSE}),
}


class ExtractedEntity(BaseModel):
    """An entity named in a passage."""

    name: str = Field(description="Full canonical name, e.g. 'Nicolas Sarkozy', not 'Sarkozy'")
    type: EntityType


class ExtractedRelation(BaseModel):
    """A relation stated in a passage, with the words that state it."""

    source: str = Field(description="Name of the source entity, exactly as in `entities`")
    type: RelationType
    target: str = Field(description="Name of the target entity, exactly as in `entities`")
    evidence: str = Field(description="Exact quote from the passage (max ~25 words) stating it")
    confidence: float = Field(ge=0, le=1, description="How explicitly the passage states it")
    date: str | None = Field(default=None, description="YYYY or YYYY-MM-DD if the passage says")
    stage: Stage | None = Field(
        default=None, description="Only for INDICTED_IN, CONVICTED_IN, ACQUITTED_IN"
    )
    final: bool | None = Field(
        default=None, description="Only for legal relations: is the decision definitive?"
    )


class Extraction(BaseModel):
    """Everything extracted from one passage."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
