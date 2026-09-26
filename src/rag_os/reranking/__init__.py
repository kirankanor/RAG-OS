"""
Importing this package registers every reranking strategy with `reranker_registry`.
"""
from rag_os.reranking import cohere_rerank, cross_encoder  # noqa: F401
from rag_os.reranking.base import Reranker, reranker_registry

__all__ = ["Reranker", "reranker_registry"]