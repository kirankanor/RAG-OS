from __future__ import annotations

import uuid

from rag_os.retrieval.pipeline_context import Candidate, PipelineContext
from rag_os.retrieval.pipeline_step import PipelineStep, pipeline_step_registry


@pipeline_step_registry.register(
    "qdrant_dense",
    "Dense vector search via Qdrant (local Docker instance or Qdrant Cloud). Persists "
    "across process restarts, unlike dense_flat/dense_hnsw which rebuild in-memory every "
    "call - closer to what you'd actually deploy. Requires the 'cloud' extra and a "
    "running Qdrant instance. NOTE: builds/uploads the collection on every run() call in "
    "this simple version - fine for experimentation, revisit before using at real scale.",
)
class QdrantDenseGenerator(PipelineStep):
    name = "qdrant_dense"
    category = "generator"

    def __init__(
        self,
        fetch_k: int = 20,
        collection_name: str = "rag_os_experiment",
        url: str = "http://localhost:6333",
        api_key: str | None = None,
    ):
        self.fetch_k = fetch_k
        self.collection_name = collection_name
        self.url = url
        self.api_key = api_key

    def run(self, ctx: PipelineContext) -> PipelineContext:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, PointStruct, VectorParams
        except ImportError as e:
            raise ImportError(
                "qdrant-client is not installed. Run: uv add --optional cloud qdrant-client"
            ) from e

        if ctx.query_vector is None or not ctx.all_chunks:
            return ctx

        client = QdrantClient(url=self.url, api_key=self.api_key)
        dim = len(ctx.query_vector)
        client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

        id_map: dict[str, str] = {}  # point_id -> chunk_id
        points = []
        for chunk in ctx.all_chunks:
            vector = ctx.all_vectors.get(chunk.id)
            if vector is None:
                continue
            point_id = str(uuid.uuid4())
            id_map[point_id] = chunk.id
            points.append(PointStruct(id=point_id, vector=vector, payload={"chunk_id": chunk.id}))
        if points:
            client.upsert(collection_name=self.collection_name, points=points)

        k = min(self.fetch_k, len(ctx.all_chunks))
        hits = client.search(collection_name=self.collection_name, query_vector=ctx.query_vector, limit=k)

        chunks_by_id = ctx.chunks_by_id()
        for hit in hits:
            chunk_id = id_map.get(str(hit.id))
            if chunk_id is None:
                continue
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            ctx.candidates.append(
                Candidate(
                    chunk=chunk,
                    vector=ctx.all_vectors.get(chunk_id),
                    score=float(hit.score),
                    source_step=self.name,
                )
            )
        return ctx
