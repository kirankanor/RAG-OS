"""LEGACY - unchanged, kept for old runs. New pipelines use generators/dense_vector.py."""
from __future__ import annotations

from rag_os.core.types import Chunk, RetrievalResult
from rag_os.retrieval.base import Retriever, retriever_registry


@retriever_registry.register(
    "faiss_flat_l2",
    "Exact (brute-force) nearest-neighbour search over a FAISS IndexFlatL2. No approximation, "
    "so it's the ground-truth baseline to compare faster/approximate indexes against. "
    "Fine up to a few hundred thousand vectors. Requires the 'local' extra.",
)
class FaissFlatRetriever(Retriever):
    name = "faiss_flat_l2"

    def __init__(self):
        self._index = None
        self._chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        try:
            import faiss
            import numpy as np
        except ImportError as e:
            raise ImportError(
                "faiss-cpu is not installed. Run: uv add --optional local faiss-cpu numpy"
            ) from e

        self._chunks = chunks
        matrix = np.array(vectors, dtype="float32")
        self._index = faiss.IndexFlatL2(matrix.shape[1])
        self._index.add(matrix)

    def retrieve(
        self, query_vector: list[float], top_k: int = 5, query_text: str = ""
    ) -> list[RetrievalResult]:
        import numpy as np

        if self._index is None:
            raise RuntimeError("Call build() before retrieve().")

        query = np.array([query_vector], dtype="float32")
        distances, indices = self._index.search(query, top_k)

        results: list[RetrievalResult] = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx == -1:
                continue
            chunk = self._chunks[idx]
            score = 1.0 / (1.0 + float(dist))
            results.append(
                RetrievalResult(
                    chunk_id=chunk.id, score=score, rank=rank, text=chunk.text,
                    metadata=chunk.metadata,
                )
            )
        return results
