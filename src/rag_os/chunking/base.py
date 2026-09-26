from __future__ import annotations

from abc import ABC, abstractmethod

from rag_os.core.registry import Registry
from rag_os.core.types import Chunk, Document


class Chunker(ABC):
    """Interface every chunking strategy must implement."""

    name: str = "base"

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a Document's text into a list of Chunks."""
        raise NotImplementedError

    def _make_chunk(
        self, document: Document, text: str, position: int, char_start: int, char_end: int
    ) -> Chunk:
        return Chunk(
            document_id=document.id,
            text=text,
            position=position,
            char_start=char_start,
            char_end=char_end,
            chunker_name=self.name,
        )


chunker_registry: Registry[Chunker] = Registry("chunker")
