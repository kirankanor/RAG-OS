from __future__ import annotations

from rag_os.chunking.base import Chunker, chunker_registry
from rag_os.core.types import Chunk, Document

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@chunker_registry.register(
    "recursive_char",
    "Tries to split on paragraph breaks first, then lines, then sentences, then words, "
    "only falling back to a hard character cut as a last resort. Good general-purpose default "
    "(same idea as LangChain's RecursiveCharacterTextSplitter).",
)
class RecursiveCharacterChunker(Chunker):
    name = "recursive_char"

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 150,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or DEFAULT_SEPARATORS

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size or not separators:
            return [text]
        sep, *rest = separators
        parts = text.split(sep) if sep else list(text)
        pieces: list[str] = []
        for part in parts:
            candidate = part if not sep else part + sep
            if len(candidate) > self.chunk_size:
                pieces.extend(self._split(candidate, rest))
            else:
                pieces.append(candidate)
        return pieces

    def _merge(self, pieces: list[str]) -> list[str]:
        merged: list[str] = []
        current = ""
        for piece in pieces:
            if len(current) + len(piece) <= self.chunk_size:
                current += piece
            else:
                if current.strip():
                    merged.append(current)
                overlap_tail = current[-self.overlap :] if self.overlap else ""
                current = overlap_tail + piece
        if current.strip():
            merged.append(current)
        return merged

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text
        pieces = self._split(text, list(self.separators))
        merged = self._merge(pieces)

        chunks: list[Chunk] = []
        cursor = 0
        for position, piece in enumerate(merged):
            start = text.find(piece.strip()[:50], cursor) if piece.strip() else cursor
            start = max(start, 0)
            end = start + len(piece)
            chunks.append(self._make_chunk(document, piece, position, start, end))
            cursor = max(cursor, end - self.overlap)
        return chunks
