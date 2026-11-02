"""Entity resolution: map the names the LLM returns to one node per real entity."""

import re
import unicodedata

from app.ingest.schema import EntityType

AFFAIR_PREFIX = re.compile(r"^(l'|les |la |le )?affaires? (du |des |de la |de l'|de |d')?")
PARENTHESIS = re.compile(r"\s*\(.*?\)\s*")


def normalize(name: str) -> str:
    """Lowercase, strip accents, unify apostrophes and collapse spaces."""
    name = unicodedata.normalize("NFKD", name.replace("’", "'"))
    name = "".join(char for char in name if not unicodedata.combining(char))
    name = PARENTHESIS.sub(" ", name.lower())
    return re.sub(r"[^\w' &-]+", " ", name).strip().replace("  ", " ")


def entity_key(name: str, entity_type: EntityType) -> str:
    """Return the unique key of an entity: its type plus its normalized name."""
    norm = normalize(name)
    if entity_type is EntityType.AFFAIR:
        norm = AFFAIR_PREFIX.sub("", norm)
    return f"{entity_type.value}:{' '.join(norm.split())}"


class Resolver:
    """Resolves names to keys, using known canonical names and last-name matching."""

    def __init__(self, canonical: dict[str, EntityType]):
        self.names: dict[str, str] = {}
        for name, entity_type in canonical.items():
            self.names[entity_key(name, entity_type)] = name

    def _surname_index(self) -> dict[str, list[str]]:
        index: dict[str, list[str]] = {}
        for key in self.names:
            if key.startswith("Person:") and " " in key:
                index.setdefault(key.rsplit(" ", 1)[1], []).append(key)
        return index

    def resolve(self, name: str, entity_type: EntityType) -> str:
        """Return the key for a name; a lone surname maps to the only person bearing it."""
        key = entity_key(name, entity_type)
        if entity_type is EntityType.PERSON and " " not in key:
            matches = self._surname_index().get(key.split(":", 1)[1], [])
            if len(matches) == 1:
                return matches[0]
        self.names.setdefault(key, name.strip())
        return key

    def display_name(self, key: str) -> str:
        """Return the name to show for a key."""
        return self.names.get(key, key.split(":", 1)[1])
