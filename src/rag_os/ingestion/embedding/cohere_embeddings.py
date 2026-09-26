from __future__ import annotations

import os

from rag_os.ingestion.embedding.base import Embedder, embedder_registry


@embedder_registry.register(
    "cohere_embed_v3",
    "Cohere embed-english-v3.0 (or multilingual-v3.0) via API. 1024-dim, supports "
    "separate input_type for documents vs queries which can meaningfully improve "
    "retrieval quality. Requires COHERE_API_KEY and the 'cloud' extra.",
)
class CohereEmbedder(Embedder):
    name = "cohere_embed_v3"

    def __init__(self, model: str = "embed-english-v3.0", api_key: str | None = None):
        try:
            import cohere
        except ImportError as e:
            raise ImportError(
                "cohere package is not installed. Run: uv add --optional cloud cohere"
            ) from e

        key = api_key or os.environ.get("COHERE_API_KEY")
        if not key:
            raise ValueError(
                "No Cohere API key found. Set COHERE_API_KEY in your environment or .env file."
            )

        self.model = model
        self._client = cohere.Client(key)
        self.dim = 1024

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embed(
            texts=texts, model=self.model, input_type="search_document"
        )
        return list(response.embeddings)

    def embed_query(self, query: str) -> list[float]:
        response = self._client.embed(
            texts=[query], model=self.model, input_type="search_query"
        )
        return next(iter(response.embeddings))
