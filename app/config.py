"""Application settings, read from the environment or a .env file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Connection settings for Neo4j and Mistral."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    mistral_api_key: str = ""
    extraction_model: str = "ministral-14b-latest"
    answer_model: str = "ministral-14b-latest"
    embedding_model: str = "mistral-embed"
    embedding_dimensions: int = 1024

    # When set, POST /api/corpus/import requires the header X-API-Key with this value.
    admin_api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings."""
    return Settings()
