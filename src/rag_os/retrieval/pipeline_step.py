from __future__ import annotations

from abc import ABC, abstractmethod

from rag_os.core.registry import Registry
from rag_os.retrieval.pipeline_context import PipelineContext


class PipelineStep(ABC):
    """One stage in a retrieval pipeline: generator, fuser, filter, expander,
    or reranker. All share this interface - a run's pipeline is just an ordered
    list of these, each independently chosen/toggled by the user.

    `category` is metadata only, for the UI to group choices sensibly; nothing
    in execution branches on it. Categories in use: "generator" (produces
    candidates), "fuser" (merges/scores candidates from multiple generators),
    "filter" (drops candidates), "expander" (attaches extra context to
    candidates), "reranker" (reorders/rescopes candidates to top_k).
    """

    name: str = "base"
    category: str = "base"

    @abstractmethod
    def run(self, ctx: PipelineContext) -> PipelineContext:
        """Read ctx, return an updated ctx."""
        raise NotImplementedError


pipeline_step_registry: Registry[PipelineStep] = Registry("pipeline_step")
