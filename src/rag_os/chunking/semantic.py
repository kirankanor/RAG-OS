from __future__ import annotations

import re
from collections.abc import Callable

from rag_os.chunking.base import Chunker, chunker_registry
from rag_os.core.types import Chunk, Document

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

EmbedFn = Callable[[list[str]], list[list[float]]]


@chunker_registry.register(
    "semantic",
    "Splits into sentences, embeds each one, and starts a new chunk wherever the "
    "cosine similarity between consecutive sentences drops below a threshold - i.e. "
    "wherever the topic seems to shift. Needs an embedding function, so it's slower "
    "and costs embedding calls, but chunk boundaries follow meaning rather than size. "
    "NOTE: pass `embed_fn` explicitly (e.g. from an Embedder strategy) when constructing this; "
    "chunking module deliberately does not import the embedding module directly.",
)
class SemanticChunker(Chunker):
    name = "semantic"

    def __init__(
        self,
        embed_fn: EmbedFn | None = None,
        similarity_threshold: float = 0.65,
        max_chunk_size: int = 2000,
    ):
        self.embed_fn = embed_fn
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def chunk(self, document: Document) -> list[Chunk]:
        if self.embed_fn is None:
            raise ValueError(
                "SemanticChunker requires an embed_fn. Construct it as "
                "SemanticChunker(embed_fn=your_embedder.embed) from the pipeline layer."
            )

        text = document.text
        sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
        if not sentences:
            return []
        if len(sentences) == 1:
            return [self._make_chunk(document, sentences[0], 0, 0, len(sentences[0]))]

        vectors = self.embed_fn(sentences)

        groups: list[list[str]] = [[sentences[0]]]
        for i in range(1, len(sentences)):
            sim = self._cosine(vectors[i - 1], vectors[i])
            current_len = sum(len(s) for s in groups[-1])
            if sim >= self.similarity_threshold and current_len < self.max_chunk_size:
                groups[-1].append(sentences[i])
            else:
                groups.append([sentences[i]])

        chunks: list[Chunk] = []
        cursor = 0
        for position, group in enumerate(groups):
            piece = " ".join(group)
            start = text.find(group[0][:30], cursor) if group[0] else cursor
            start = max(start, 0)
            end = start + len(piece)
            chunks.append(self._make_chunk(document, piece, position, start, end))
            cursor = start
        return chunks
