from __future__ import annotations

from abc import ABC, abstractmethod

from rag_os.core.registry import Registry
from rag_os.core.types import RetrievalResult


class Reranker(ABC):
    """Interface every reranking strategy must implement. Takes retriever's top-N
    candidates and re-scores them; caller then keeps the top-K after reranking."""

    name: str = "base"

    @abstractmethod
    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int = 5
    ) -> list[RetrievalResult]:
        raise NotImplementedError


reranker_registry: Registry[Reranker] = Registry("reranker")