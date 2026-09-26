from __future__ import annotations

from collections import defaultdict

from rag_os.retrieval.pipeline_context import Candidate, PipelineContext
from rag_os.retrieval.pipeline_step import PipelineStep, pipeline_step_registry


def _group_by_chunk(candidates: list[Candidate]) -> dict[str, list[Candidate]]:
    groups: dict[str, list[Candidate]] = defaultdict(list)
    for c in candidates:
        groups[c.chunk.id].append(c)
    return groups


@pipeline_step_registry.register(
    "rrf_fuse",
    "Reciprocal Rank Fusion. Merges candidates from multiple generators (e.g. dense_flat "
    "+ bm25) by rank position rather than raw score, so it needs no score-scale "
    "normalization between generators - the fusion method the course specifically "
    "recommends over a weighted blend. Run this after 2+ generator steps, before any "
    "filter/expander/reranker.",
)
class RrfFuser(PipelineStep):
    name = "rrf_fuse"
    category = "fuser"

    def __init__(self, k: int = 60):
        self.k = k  # RRF damping constant - higher = flatter weighting across ranks

    def run(self, ctx: PipelineContext) -> PipelineContext:
        groups = _group_by_chunk(ctx.candidates)
        if not groups:
            return ctx

        # Rank each generator's own candidate list independently before fusing,
        # so fusion only needs positions, never raw score magnitudes.
        by_source: dict[str, list[Candidate]] = defaultdict(list)
        for c in ctx.candidates:
            by_source[c.source_step].append(c)
        for source_candidates in by_source.values():
            source_candidates.sort(key=lambda c: c.score, reverse=True)

        rrf_scores: dict[str, float] = defaultdict(float)
        for source_candidates in by_source.values():
            for rank, c in enumerate(source_candidates):
                rrf_scores[c.chunk.id] += 1.0 / (self.k + rank + 1)

        fused: list[Candidate] = []
        for chunk_id, group in groups.items():
            best = group[0]
            fused.append(
                Candidate(
                    chunk=best.chunk,
                    vector=best.vector,
                    score=rrf_scores[chunk_id],
                    source_step=self.name,
                    parent_text=best.parent_text,
                )
            )
        fused.sort(key=lambda c: c.score, reverse=True)
        ctx.candidates = fused
        return ctx


@pipeline_step_registry.register(
    "weighted_fuse",
    "Merges candidates from multiple generators by min-max normalizing each generator's "
    "scores to [0,1], then blending with a configurable weight. Simpler than RRF but "
    "sensitive to each generator's score distribution - prefer rrf_fuse unless you have "
    "a specific reason to weight one generator's confidence directly.",
)
class WeightedFuser(PipelineStep):
    name = "weighted_fuse"
    category = "fuser"

    def __init__(self, weights: dict[str, float] | None = None):
        # weights keyed by generator step name, e.g. {"dense_flat": 0.5, "bm25": 0.5}.
        # Missing sources default to equal weight.
        self.weights = weights or {}

    @staticmethod
    def _normalize(scores: list[float]) -> list[float]:
        if not scores:
            return scores
        lo, hi = min(scores), max(scores)
        if hi - lo < 1e-9:
            return [0.0 for _ in scores]
        return [(s - lo) / (hi - lo) for s in scores]

    def run(self, ctx: PipelineContext) -> PipelineContext:
        by_source: dict[str, list[Candidate]] = defaultdict(list)
        for c in ctx.candidates:
            by_source[c.source_step].append(c)
        if not by_source:
            return ctx

        default_weight = 1.0 / len(by_source)
        normalized_scores: dict[str, dict[str, float]] = {}  # source -> chunk_id -> norm score
        for source, cands in by_source.items():
            norm = self._normalize([c.score for c in cands])
            normalized_scores[source] = {c.chunk.id: n for c, n in zip(cands, norm)}

        combined: dict[str, float] = defaultdict(float)
        chunk_lookup: dict[str, Candidate] = {}
        for source, cands in by_source.items():
            weight = self.weights.get(source, default_weight)
            for c in cands:
                combined[c.chunk.id] += weight * normalized_scores[source][c.chunk.id]
                chunk_lookup[c.chunk.id] = c

        fused: list[Candidate] = []
        for chunk_id, score in combined.items():
            base = chunk_lookup[chunk_id]
            fused.append(
                Candidate(
                    chunk=base.chunk, vector=base.vector, score=score,
                    source_step=self.name, parent_text=base.parent_text,
                )
            )
        fused.sort(key=lambda c: c.score, reverse=True)
        ctx.candidates = fused
        return ctx
