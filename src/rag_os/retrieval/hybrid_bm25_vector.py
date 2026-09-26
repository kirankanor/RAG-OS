"""LEGACY - unchanged (includes the RRF fusion_method added earlier this session),
kept for old runs. New pipelines split this into generators/bm25.py + 
generators/dense_vector.py + fusers/fuse.py, composed independently."""
from __future__ import annotations

from rag_os.core.types import Chunk, RetrievalResult
from rag_os.retrieval.base import Retriever, retriever_registry


@retriever_registry.register(
    "hybrid_bm25_vector",
    "Combines lexical BM25 scoring (good for exact keyword/name matches) with dense "
    "vector similarity (good for paraphrase/semantic matches) via a weighted score fusion "
    "or reciprocal rank fusion (RRF). Often beats either alone. Requires the 'local' "
    "extra (rank-bm25 + numpy).",
)
class HybridBm25VectorRetriever(Retriever):
    name = "hybrid_bm25_vector"

    def __init__(self, alpha: float = 0.5, fusion_method: str = "weighted", rrf_k: int = 60):
        """alpha: weight on vector score in 'weighted' fusion; ignored for 'rrf'.
        fusion_method: 'weighted' (score blend) or 'rrf' (reciprocal rank fusion,
        recommended when BM25 and vector scores aren't on comparable scales)."""
        self.alpha = alpha
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
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

    @staticmethod
    def _ranks(scores: list[float]) -> list[int]:
        """Rank 0 = highest score."""
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ranks = [0] * len(scores)
        for r, i in enumerate(order):
            ranks[i] = r
        return ranks

    def retrieve(
        self, query_vector: list[float], top_k: int = 5, query_text: str = ""
    ) -> list[RetrievalResult]:
        if self._bm25 is None:
            raise RuntimeError("Call build() before retrieve().")

        bm25_raw = list(self._bm25.get_scores(query_text.lower().split()))
        vector_raw = [self._cosine(query_vector, v) for v in self._vectors]

        if self.fusion_method == "rrf":
            bm25_ranks = self._ranks(bm25_raw)
            vector_ranks = self._ranks(vector_raw)
            combined = [
                1.0 / (self.rrf_k + bm25_ranks[i]) + 1.0 / (self.rrf_k + vector_ranks[i])
                for i in range(len(self._chunks))
            ]
        else:
            bm25_scores = self._normalize(bm25_raw)
            vector_scores = self._normalize(vector_raw)
            combined = [
                self.alpha * v + (1 - self.alpha) * b
                for v, b in zip(vector_scores, bm25_scores)
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
