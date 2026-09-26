from __future__ import annotations

from abc import ABC, abstractmethod

from rag_os.core.registry import Registry


class Embedder(ABC):
    """Interface every embedding strategy must implement."""

    name: str = "base"
    dim: int = 0

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per input text, same order."""
        raise NotImplementedError

    def embed_query(self, query: str) -> list[float]:
        """Some providers use a different mode/prefix for queries vs documents; default
        implementation just reuses embed() for a single-item batch."""
        return self.embed([query])[0]


embedder_registry: Registry[Embedder] = Registry("embedder")
