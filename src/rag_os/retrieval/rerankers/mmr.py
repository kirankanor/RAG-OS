from __future__ import annotations

from rag_os.retrieval.pipeline_context import PipelineContext
from rag_os.retrieval.pipeline_step import PipelineStep, pipeline_step_registry


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


@pipeline_step_registry.register(
    "mmr",
    "Maximal Marginal Relevance reranking. Greedily picks candidates that balance "
    "relevance (score) against novelty (low similarity to already-picked candidates), "
    "so top_k isn't several near-duplicate chunks of the same paragraph. Needs each "
    "Candidate.vector to already be populated (every generator in this package sets it) "
    "- run this as the last reranker step, after any cross-encoder/Cohere reranking if "
    "you're using both.",
)
class MmrReranker(PipelineStep):
    name = "mmr"
    category = "reranker"

    def __init__(self, lambda_mult: float = 0.5):
        """lambda_mult: 1.0 = pure relevance (no diversity), 0.0 = pure diversity
        (ignores relevance). 0.5 = balanced."""
        self.lambda_mult = lambda_mult

    def run(self, ctx: PipelineContext) -> PipelineContext:
        pool = [c for c in ctx.candidates if c.vector is not None]
        if not pool:
            return ctx

        max_score = max(c.score for c in pool) or 1.0
        relevance = {id(c): c.score / max_score for c in pool}

        selected = []
        remaining = list(pool)

        while remaining and len(selected) < ctx.top_k:
            best_candidate = None
            best_mmr = float("-inf")
            for c in remaining:
                if selected:
                    max_sim = max(_cosine(c.vector, s.vector) for s in selected)
                else:
                    max_sim = 0.0
                mmr = self.lambda_mult * relevance[id(c)] - (1 - self.lambda_mult) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_candidate = c
            selected.append(best_candidate)
            remaining.remove(best_candidate)

        ctx.candidates = selected
        return ctx
