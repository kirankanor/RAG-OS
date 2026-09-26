from __future__ import annotations

from rag_os.embedding.base import Embedder, embedder_registry


@embedder_registry.register(
    "local_minilm",
    "Local sentence-transformers model (all-MiniLM-L6-v2 by default). Free, runs offline, "
    "384-dim, fast on CPU. Good default when you don't want to spend on API calls. "
    "Requires the 'local' extra.",
)
class LocalMiniLmEmbedder(Embedder):
    name = "local_minilm"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: uv add --optional local sentence-transformers"
            ) from e

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return vectors.tolist()
