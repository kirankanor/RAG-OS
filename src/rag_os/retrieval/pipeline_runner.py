"""
Assembles and runs a retrieval pipeline: an ordered list of (step_name, params)
pairs, each resolved against `pipeline_step_registry` and run in sequence against
one shared PipelineContext. This is the Path-B replacement for the old single
`Retriever.retrieve()` call - a run's config is now a list of steps instead of
one retriever_name + one reranker_name.
"""
from __future__ import annotations

from rag_os.core.types import Chunk, RetrievalResult
from rag_os.retrieval.pipeline_context import Candidate, PipelineContext
from rag_os.retrieval.pipeline_step import pipeline_step_registry

# Import every step package so their decorators run and register with pipeline_step_registry.
from rag_os.retrieval import expanders, filters, fusers, generators, rerankers  # noqa: F401


class PipelineStepConfig:
    """One entry in a run's assembled pipeline: which step, with what params."""

    def __init__(self, step_name: str, params: dict | None = None):
        self.step_name = step_name
        self.params = params or {}


def run_pipeline(
    step_configs: list[PipelineStepConfig],
    query: str,
    query_vector: list[float] | None,
    all_chunks: list[Chunk],
    all_vectors: dict[str, list[float]],
    top_k: int = 5,
) -> list[RetrievalResult]:
    """Builds a PipelineContext, runs every configured step against it in order,
    and converts the final candidates into RetrievalResults for display/storage -
    the same output type app/pages/5_Retrieval.py already expects."""
    ctx = PipelineContext(
        query=query,
        query_vector=query_vector,
        all_chunks=all_chunks,
        all_vectors=all_vectors,
        top_k=top_k,
    )

    for step_config in step_configs:
        step = pipeline_step_registry.create(step_config.step_name, **step_config.params)
        ctx = step.run(ctx)

    # If no reranker step trimmed to top_k (e.g. a generator-only pipeline with
    # no fuser/reranker), sort by score and trim here as a fallback.
    if len(ctx.candidates) > top_k:
        ctx.candidates = sorted(ctx.candidates, key=lambda c: c.score, reverse=True)[:top_k]

    return _candidates_to_results(ctx.candidates)


def _candidates_to_results(candidates: list[Candidate]) -> list[RetrievalResult]:
    results = []
    for rank, c in enumerate(candidates):
        metadata = dict(c.chunk.metadata)
        if c.parent_text:
            metadata["expanded_context"] = c.parent_text
        results.append(
            RetrievalResult(chunk_id=c.chunk.id, score=c.score, rank=rank, text=c.chunk.text, metadata=metadata)
        )
    return results
