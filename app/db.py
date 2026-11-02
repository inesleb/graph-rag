"""Neo4j driver access."""

from functools import lru_cache

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from app.config import get_settings


@lru_cache
def get_driver() -> Driver:
    """Return the shared Neo4j driver."""
    settings = get_settings()
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
        # Typed-relation unions name every RelationType, even ones absent from a small graph;
        # "unknown relationship type" notices are expected there and only add noise.
        notifications_disabled_classifications=["UNRECOGNIZED"],
    )


def check_connection() -> bool:
    """Return True when Neo4j answers."""
    try:
        get_driver().verify_connectivity()
    except (Neo4jError, ServiceUnavailable, OSError, ValueError):
        return False
    return True
