"""
LEGACY - kept unchanged so existing runs (created before the Path-B pipeline
refactor) still load via run_manager.load_retriever_for_run(). New runs should
use retrieval/pipeline_runner.py + retrieval/pipeline_step.py instead.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from rag_os.core.registry import Registry
from rag_os.core.types import Chunk, RetrievalResult


class Retriever(ABC):
    """Interface every (legacy) retrieval strategy must implement.

    Usage pattern: build() once per run (with the chunks + their embeddings for that run),
    then retrieve() as many times as you want with different queries.
    """

    name: str = "base"

    @abstractmethod
    def build(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Index a set of chunks + their precomputed vectors."""
        raise NotImplementedError

    @abstractmethod
    def retrieve(
        self, query_vector: list[float], top_k: int = 5, query_text: str = ""
    ) -> list[RetrievalResult]:
        """Return the top_k most relevant chunks for a query.

        query_text is optional and only used by lexical/hybrid strategies (e.g. BM25)
        that need the raw query string in addition to (or instead of) its vector.
        """
        raise NotImplementedError


retriever_registry: Registry[Retriever] = Registry("retriever")
