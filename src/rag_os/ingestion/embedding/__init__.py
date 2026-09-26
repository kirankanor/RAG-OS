"""
Importing this package registers every embedding strategy with `embedder_registry`.
"""
from rag_os.ingestion.embedding import (  # noqa: F401
    cohere_embeddings,
    local_sentence_transformers,
    openai_embeddings,
)
from rag_os.ingestion.embedding.base import Embedder, embedder_registry

__all__ = ["Embedder", "embedder_registry"]
