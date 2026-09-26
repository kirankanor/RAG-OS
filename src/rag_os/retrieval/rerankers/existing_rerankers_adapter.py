"""
Thin PipelineStep wrappers around the (legacy) Reranker classes in this same
package (cross_encoder.py, cohere_rerank.py). Kept as adapters rather than
merged into one implementation, so the Reranker interface (still used by the
pre-Path-B retriever_name/reranker_name flow) keeps working unmodified.
"""
from __future__ import annotations

from rag_os.core.types import RetrievalResult
from rag_os.retrieval.pipeline_context import Candidate, PipelineContext
from rag_os.retrieval.pipeline_step import PipelineStep, pipeline_step_registry


def _candidates_to_results(candidates: list[Candidate]) -> list[RetrievalResult]:
    return [
        RetrievalResult(chunk_id=c.chunk.id, score=c.score, rank=i, text=c.chunk.text, metadata=c.chunk.metadata)
        for i, c in enumerate(candidates)
    ]


def _reorder_candidates(candidates: list[Candidate], reranked_results: list[RetrievalResult]) -> list[Candidate]:
    by_chunk_id = {c.chunk.id: c for c in candidates}
    ordered = []
    for r in reranked_results:
        c = by_chunk_id.get(r.chunk_id)
        if c is not None:
            c.score = r.score
            ordered.append(c)
    return ordered


@pipeline_step_registry.register(
    "cross_encoder_rerank",
    "Pipeline-step wrapper around retrieval.rerankers.cross_encoder.CrossEncoderReranker "
    "(local sentence-transformers cross-encoder). Requires the 'local' extra.",
)
class CrossEncoderRerankStep(PipelineStep):
    name = "cross_encoder_rerank"
    category = "reranker"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from rag_os.retrieval.rerankers.cross_encoder import CrossEncoderReranker

        self._reranker = CrossEncoderReranker(model_name=model_name)

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.candidates:
            return ctx
        results = _candidates_to_results(ctx.candidates)
        reranked = self._reranker.rerank(ctx.query, results, top_k=ctx.top_k)
        ctx.candidates = _reorder_candidates(ctx.candidates, reranked)
        return ctx


@pipeline_step_registry.register(
    "cohere_rerank",
    "Pipeline-step wrapper around retrieval.rerankers.cohere_rerank.CohereReranker "
    "(Cohere rerank API). Requires COHERE_API_KEY and the 'cloud' extra.",
)
class CohereRerankStep(PipelineStep):
    name = "cohere_rerank"
    category = "reranker"

    def __init__(self, model: str = "rerank-english-v3.0", api_key: str | None = None):
        from rag_os.retrieval.rerankers.cohere_rerank import CohereReranker

        self._reranker = CohereReranker(model=model, api_key=api_key)

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.candidates:
            return ctx
        results = _candidates_to_results(ctx.candidates)
        reranked = self._reranker.rerank(ctx.query, results, top_k=ctx.top_k)
        ctx.candidates = _reorder_candidates(ctx.candidates, reranked)
        return ctx
