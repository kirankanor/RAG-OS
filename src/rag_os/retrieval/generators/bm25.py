from __future__ import annotations

from rag_os.retrieval.pipeline_context import Candidate, PipelineContext
from rag_os.retrieval.pipeline_step import PipelineStep, pipeline_step_registry


@pipeline_step_registry.register(
    "bm25",
    "Lexical (keyword) search via BM25. Strong for exact terms, names, IDs, acronyms "
    "that dense embeddings can blur together. Pair with a dense generator + a fuser "
    "step (rrf_fuse or weighted_fuse) for hybrid search. Requires the 'local' extra "
    "(rank-bm25).",
)
class Bm25Generator(PipelineStep):
    name = "bm25"
    category = "generator"

    def __init__(self, fetch_k: int = 20):
        self.fetch_k = fetch_k

    def run(self, ctx: PipelineContext) -> PipelineContext:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise ImportError(
                "rank-bm25 is not installed. Run: uv add --optional local rank-bm25"
            ) from e

        if not ctx.all_chunks or not ctx.query.strip():
            return ctx

        tokenized_corpus = [c.text.lower().split() for c in ctx.all_chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(ctx.query.lower().split())

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        k = min(self.fetch_k, len(ctx.all_chunks))

        for idx, score in ranked[:k]:
            chunk = ctx.all_chunks[idx]
            ctx.candidates.append(
                Candidate(
                    chunk=chunk,
                    vector=ctx.all_vectors.get(chunk.id),
                    score=float(score),
                    source_step=self.name,
                )
            )
        return ctx
