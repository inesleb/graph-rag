"""Request and response models of the API."""

from typing import Literal

from pydantic import BaseModel, Field


class Source(BaseModel):
    """A passage, relation or path the answer relies on."""

    id: str
    kind: Literal["passage", "relation", "path"]
    text: str
    document: str | None = None
    section: str | None = None
    url: str | None = None
    evidence: list[str] | None = None
    chunk_ids: list[str] | None = None


class ToolCall(BaseModel):
    """One step of the agent, for transparency."""

    tool: str
    args: dict
    results: int


class AskResponse(BaseModel):
    """Answer of the GraphRAG agent."""

    question: str
    status: Literal["answered", "not_in_corpus"]
    answer: str
    confidence: float
    citations: list[str]
    sources: list[Source]
    graph_paths: list[str]
    entities: list[dict]
    tool_calls: list[ToolCall]
    guardrail: str | None = None


class ImportRequest(BaseModel):
    """Articles to add to the corpus. With no titles, the existing corpus is re-ingested."""

    titles: list[str] = Field(default_factory=list, max_length=20)
    kind: Literal["affair", "context"] = "affair"
    entity_type: Literal[
        "Affair", "Person", "Party", "Company", "Institution", "Media"
    ] = "Affair"
    use_llm: bool = Field(True, description="False: only use cached extractions")
