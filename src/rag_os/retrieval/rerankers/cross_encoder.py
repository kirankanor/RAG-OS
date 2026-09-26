from __future__ import annotations

from rag_os.core.types import RetrievalResult
from rag_os.retrieval.rerankers.base import Reranker, reranker_registry


@reranker_registry.register(
    "cross_encoder",
    "Local cross-encoder reranker (ms-marco-MiniLM by default) via sentence-transformers. "
    "Free, runs offline, scores each (query, chunk) pair jointly rather than via cosine "
    "similarity of separate embeddings - typically more accurate than the retriever's own "
    "ranking. Requires the 'local' extra.",
)
class CrossEncoderReranker(Reranker):
    name = "cross_encoder"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed. Run: uv add --optional local sentence-transformers"
            ) from e
        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int = 5
    ) -> list[RetrievalResult]:
        if not results:
            return results
        pairs = [(query, r.text) for r in results]
        scores = self._model.predict(pairs)
        reranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)[:top_k]
        out = []
        for rank, (r, score) in enumerate(reranked):
            out.append(RetrievalResult(chunk_id=r.chunk_id, score=float(score), rank=rank, text=r.text, metadata=r.metadata))
        return out
