"""Mistral clients for chat, structured extraction and embeddings."""

from functools import lru_cache

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

from app.config import get_settings


@lru_cache
def get_chat_model(purpose: str = "answer") -> ChatMistralAI:
    """Return the chat model used for extraction or for answers."""
    settings = get_settings()
    model = settings.extraction_model if purpose == "extraction" else settings.answer_model
    return ChatMistralAI(
        model=model,
        api_key=settings.mistral_api_key,
        temperature=0,
        max_retries=6,
        timeout=120,
    )


@lru_cache
def get_embeddings() -> MistralAIEmbeddings:
    """Return the embedding model."""
    settings = get_settings()
    return MistralAIEmbeddings(
        model=settings.embedding_model, api_key=settings.mistral_api_key, max_retries=6
    )
