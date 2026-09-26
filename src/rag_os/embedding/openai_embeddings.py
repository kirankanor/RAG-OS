from __future__ import annotations

import os

from rag_os.embedding.base import Embedder, embedder_registry


@embedder_registry.register(
    "openai_text_embedding_3_small",
    "OpenAI's text-embedding-3-small via API. 1536-dim, cheap, strong general-purpose "
    "quality. Requires OPENAI_API_KEY and the 'cloud' extra.",
)
class OpenAiEmbedder(Embedder):
    name = "openai_text_embedding_3_small"

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package is not installed. Run: uv add --optional cloud openai"
            ) from e

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "No OpenAI API key found. Set OPENAI_API_KEY in your environment or .env file."
            )

        self.model = model
        self._client = OpenAI(api_key=key)
        self.dim = 1536 if model == "text-embedding-3-small" else 3072

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]
