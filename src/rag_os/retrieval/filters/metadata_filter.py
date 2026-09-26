from __future__ import annotations

from typing import Any

from rag_os.retrieval.pipeline_context import PipelineContext
from rag_os.retrieval.pipeline_step import PipelineStep, pipeline_step_registry


@pipeline_step_registry.register(
    "metadata_filter",
    "Drops candidates whose Chunk.metadata doesn't match the given filters, before "
    "reranking/expansion. Supports exact-match filters and range filters. Run this "
    "right after fusion (or right after a single generator, if not using fusion), "
    "before any reranker step.",
)
class MetadataFilter(PipelineStep):
    name = "metadata_filter"
    category = "filter"

    def __init__(
        self,
        equals: dict[str, Any] | None = None,
        gte: dict[str, Any] | None = None,
        lte: dict[str, Any] | None = None,
    ):
        """
        equals: {"source_filename": "handbook.pdf"} - candidate.chunk.metadata[key] == value
        gte / lte: {"page_number": 3} - candidate.chunk.metadata[key] >= / <= value

        All three dicts are ANDed together; missing keys on a chunk fail the filter
        (chunk is dropped) rather than passing by default, since a missing field
        usually means the filter doesn't apply to that chunk's content type.
        """
        self.equals = equals or {}
        self.gte = gte or {}
        self.lte = lte or {}

    def _passes(self, metadata: dict[str, Any]) -> bool:
        for key, value in self.equals.items():
            if metadata.get(key) != value:
                return False
        for key, value in self.gte.items():
            if key not in metadata or metadata[key] < value:
                return False
        for key, value in self.lte.items():
            if key not in metadata or metadata[key] > value:
                return False
        return True

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if not (self.equals or self.gte or self.lte):
            return ctx
        ctx.candidates = [c for c in ctx.candidates if self._passes(c.chunk.metadata)]
        return ctx
