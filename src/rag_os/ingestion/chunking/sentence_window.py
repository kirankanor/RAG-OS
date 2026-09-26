from __future__ import annotations

import re

from rag_os.ingestion.chunking.base import Chunker, chunker_registry
from rag_os.core.types import Chunk, Document

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@chunker_registry.register(
    "sentence_window",
    "Splits into sentences, then groups N sentences per chunk with a configurable "
    "sentence overlap between consecutive chunks. Keeps chunk boundaries semantically "
    "cleaner than a raw character cut.",
)
class SentenceWindowChunker(Chunker):
    name = "sentence_window"

    def __init__(self, sentences_per_chunk: int = 5, sentence_overlap: int = 1):
        self.sentences_per_chunk = max(1, sentences_per_chunk)
        self.sentence_overlap = max(0, min(sentence_overlap, sentences_per_chunk - 1))

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text
        sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
        if not sentences:
            return []

        step = self.sentences_per_chunk - self.sentence_overlap
        chunks: list[Chunk] = []
        position = 0
        i = 0
        cursor = 0
        while i < len(sentences):
            window = sentences[i : i + self.sentences_per_chunk]
            piece = " ".join(window)
            start = text.find(window[0][:30], cursor) if window[0] else cursor
            start = max(start, 0)
            end = start + len(piece)
            chunks.append(self._make_chunk(document, piece, position, start, end))
            cursor = start
            position += 1
            i += step
        return chunks
