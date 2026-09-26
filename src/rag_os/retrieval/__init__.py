"""
Importing this package registers every retrieval strategy with `retriever_registry`.
"""
from rag_os.retrieval import (  # noqa: F401
    faiss_local,
    hybrid_bm25_vector,
    qdrant_cloud,
)
from rag_os.retrieval.base import Retriever, retriever_registry

__all__ = ["Retriever", "retriever_registry"]
