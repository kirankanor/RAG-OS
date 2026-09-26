from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag_os.core.types import Chunk


@dataclass
class Candidate:
    """One chunk moving through the pipeline. Carries the full Chunk + its vector
    (not just text/score like the old RetrievalResult) because later steps
    (MMR, expanders) need more than text to do their job."""

    chunk: Chunk
    vector: list[float] | None = None
    score: float = 0.0
    source_step: str = ""  # which step last set this score, useful for debugging
    parent_text: str = ""  # populated by expander steps (parent-document / sentence-window)


@dataclass
class PipelineContext:
    """Threaded through every step in a retrieval pipeline run, in order. Each
    step reads from this and returns an updated version (mutated in place or
    replaced - callers don't care which)."""

    query: str
    query_vector: list[float] | None
    all_chunks: list[Chunk]  # full indexed set for this run, for generators to search
    all_vectors: dict[str, list[float]]  # chunk_id -> vector, for generators + MMR
    candidates: list[Candidate] = field(default_factory=list)
    top_k: int = 5
    scratch: dict[str, Any] = field(default_factory=dict)  # step-to-step handoff data

    def chunks_by_id(self) -> dict[str, Chunk]:
        return {c.id: c for c in self.all_chunks}
