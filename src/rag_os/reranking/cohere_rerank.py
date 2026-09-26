from __future__ import annotations

import os

from rag_os.core.types import RetrievalResult
from rag_os.reranking.base import Reranker, reranker_registry


@reranker_registry.register(
    "cohere_rerank",
    "Cohere's rerank API (rerank-english-v3.0). Cloud call, no local model needed, "
    "often stronger than a local cross-encoder. Requires COHERE_API_KEY and the 'cloud' extra.",
)
class CohereReranker(Reranker):
    name = "cohere_rerank"

    def __init__(self, model: str = "rerank-english-v3.0", api_key: str | None = None):
        try:
            import cohere
        except ImportError as e:
            raise ImportError("cohere package is not installed. Run: uv add --optional cloud cohere") from e
        key = api_key or os.environ.get("COHERE_API_KEY")
        if not key:
            raise ValueError("No Cohere API key found. Set COHERE_API_KEY in your environment or .env file.")
        self.model = model
        self._client = cohere.Client(key)

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int = 5
    ) -> list[RetrievalResult]:
        if not results:
            return results
        response = self._client.rerank(
            query=query, documents=[r.text for r in results], top_n=top_k, model=self.model
        )
        out = []
        for rank, item in enumerate(response.results):
            r = results[item.index]
            out.append(RetrievalResult(chunk_id=r.chunk_id, score=float(item.relevance_score), rank=rank, text=r.text, metadata=r.metadata))
        return out