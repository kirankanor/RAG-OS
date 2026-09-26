from __future__ import annotations

from rag_os.ingestion.chunking.base import Chunker, chunker_registry
from rag_os.core.types import Chunk, Document


@chunker_registry.register(
    "fixed_size",
    "Splits text into fixed-size character windows with optional overlap. "
    "Simplest possible baseline - ignores sentence/paragraph boundaries entirely.",
)
class FixedSizeChunker(Chunker):
    name = "fixed_size"

    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text
        step = max(1, self.chunk_size - self.overlap)
        chunks: list[Chunk] = []
        position = 0
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            piece = text[start:end]
            if piece.strip():
                chunks.append(self._make_chunk(document, piece, position, start, end))
                position += 1
            if end == len(text):
                break
            start += step
        return chunks
