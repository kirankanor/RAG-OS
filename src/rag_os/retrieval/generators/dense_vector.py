from __future__ import annotations

from rag_os.retrieval.pipeline_context import Candidate, PipelineContext
from rag_os.retrieval.pipeline_step import PipelineStep, pipeline_step_registry


@pipeline_step_registry.register(
    "dense_flat",
    "Exact (brute-force) dense vector similarity search via FAISS IndexFlatL2. "
    "No approximation - the ground-truth baseline to compare faster/approximate "
    "generators against. Fine up to a few hundred thousand vectors. "
    "Requires the 'local' extra.",
)
class DenseFlatGenerator(PipelineStep):
    name = "dense_flat"
    category = "generator"

    def __init__(self, fetch_k: int = 20):
        self.fetch_k = fetch_k  # how many candidates to pull before later steps trim to top_k

    def run(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import faiss
            import numpy as np
        except ImportError as e:
            raise ImportError(
                "faiss-cpu is not installed. Run: uv add --optional local faiss-cpu numpy"
            ) from e

        if ctx.query_vector is None or not ctx.all_chunks:
            return ctx

        chunk_ids = [c.id for c in ctx.all_chunks]
        matrix = np.array([ctx.all_vectors[cid] for cid in chunk_ids], dtype="float32")
        index = faiss.IndexFlatL2(matrix.shape[1])
        index.add(matrix)

        query = np.array([ctx.query_vector], dtype="float32")
        k = min(self.fetch_k, len(chunk_ids))
        distances, indices = index.search(query, k)

        chunks_by_id = ctx.chunks_by_id()
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            cid = chunk_ids[idx]
            chunk = chunks_by_id[cid]
            score = 1.0 / (1.0 + float(dist))
            ctx.candidates.append(
                Candidate(chunk=chunk, vector=ctx.all_vectors.get(cid), score=score, source_step=self.name)
            )
        return ctx


@pipeline_step_registry.register(
    "dense_hnsw",
    "Approximate dense vector similarity search via FAISS IndexHNSWFlat. Sub-linear "
    "query time vs. dense_flat's brute force - use once your chunk count grows past "
    "roughly 100k. Slightly lower recall than exact search in exchange for speed. "
    "Requires the 'local' extra.",
)
class DenseHnswGenerator(PipelineStep):
    name = "dense_hnsw"
    category = "generator"

    def __init__(self, fetch_k: int = 20, m: int = 32, ef_search: int = 64):
        self.fetch_k = fetch_k
        self.m = m  # HNSW graph connectivity - higher = better recall, more memory
        self.ef_search = ef_search  # search-time candidate list size - higher = better recall, slower

    def run(self, ctx: PipelineContext) -> PipelineContext:
        try:
            import faiss
            import numpy as np
        except ImportError as e:
            raise ImportError(
                "faiss-cpu is not installed. Run: uv add --optional local faiss-cpu numpy"
            ) from e

        if ctx.query_vector is None or not ctx.all_chunks:
            return ctx

        chunk_ids = [c.id for c in ctx.all_chunks]
        matrix = np.array([ctx.all_vectors[cid] for cid in chunk_ids], dtype="float32")

        index = faiss.IndexHNSWFlat(matrix.shape[1], self.m)
        index.hnsw.efSearch = self.ef_search
        index.add(matrix)

        query = np.array([ctx.query_vector], dtype="float32")
        k = min(self.fetch_k, len(chunk_ids))
        distances, indices = index.search(query, k)

        chunks_by_id = ctx.chunks_by_id()
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            cid = chunk_ids[idx]
            chunk = chunks_by_id[cid]
            score = 1.0 / (1.0 + float(dist))
            ctx.candidates.append(
                Candidate(chunk=chunk, vector=ctx.all_vectors.get(cid), score=score, source_step=self.name)
            )
        return ctx
