from __future__ import annotations

from rag_os.core.types import Chunk, RetrievalResult
from rag_os.retrieval.base import Retriever, retriever_registry


@retriever_registry.register(
    "hybrid_bm25_vector",
    "Combines lexical BM25 scoring (good for exact keyword/name matches) with dense "
    "vector similarity (good for paraphrase/semantic matches) via a weighted score fusion. "
    "Often beats either alone. Requires the 'local' extra (rank-bm25 + numpy).",
)
class HybridBm25VectorRetriever(Retriever):
    name = "hybrid_bm25_vector"

    def __init__(self, alpha: float = 0.5):
        """alpha: weight on the vector score; (1 - alpha) goes to BM25. 0.5 = equal blend."""
        self.alpha = alpha
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []
        self._bm25 = None

    def build(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise ImportError(
                "rank-bm25 is not installed. Run: uv add --optional local rank-bm25"
            ) from e

        self._chunks = chunks
        self._vectors = vectors
        tokenized = [c.text.lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenized)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    @staticmethod
    def _normalize(scores: list[float]) -> list[float]:
        if not scores:
            return scores
        lo, hi = min(scores), max(scores)
        if hi - lo < 1e-9:
            return [0.0 for _ in scores]
        return [(s - lo) / (hi - lo) for s in scores]

    def retrieve(
        self, query_vector: list[float], top_k: int = 5, query_text: str = ""
    ) -> list[RetrievalResult]:
        if self._bm25 is None:
            raise RuntimeError("Call build() before retrieve().")

        bm25_scores = self._normalize(list(self._bm25.get_scores(query_text.lower().split())))
        vector_scores = self._normalize(
            [self._cosine(query_vector, v) for v in self._vectors]
        )

        combined = [
            self.alpha * v_score + (1 - self.alpha) * b_score
            for v_score, b_score in zip(vector_scores, bm25_scores)
        ]

        ranked = sorted(enumerate(combined), key=lambda x: x[1], reverse=True)[:top_k]
        results: list[RetrievalResult] = []
        for rank, (idx, score) in enumerate(ranked):
            chunk = self._chunks[idx]
            results.append(
                RetrievalResult(
                    chunk_id=chunk.id, score=float(score), rank=rank, text=chunk.text,
                    metadata=chunk.metadata,
                )
            )
        return results
