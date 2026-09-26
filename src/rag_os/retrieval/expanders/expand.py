from __future__ import annotations

from rag_os.retrieval.pipeline_context import PipelineContext
from rag_os.retrieval.pipeline_step import PipelineStep, pipeline_step_registry


@pipeline_step_registry.register(
    "parent_document_expand",
    "Attaches each candidate's parent chunk text as extra context (Candidate.parent_text), "
    "via Chunk.parent_chunk_id. Matching still happened on the small chunk (precise), but "
    "the LLM sees the larger parent (more context). Requires chunks to have been built with "
    "parent_chunk_id populated - if your chunker doesn't set it, this step is a no-op. "
    "Run near the end of the pipeline, after filtering/reranking has picked the final set.",
)
class ParentDocumentExpander(PipelineStep):
    name = "parent_document_expand"
    category = "expander"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        chunks_by_id = ctx.chunks_by_id()
        for candidate in ctx.candidates:
            parent_id = candidate.chunk.parent_chunk_id
            if not parent_id:
                continue
            parent = chunks_by_id.get(parent_id)
            if parent is not None:
                candidate.parent_text = parent.text
        return ctx


@pipeline_step_registry.register(
    "sentence_window_expand",
    "Attaches N neighboring chunks (by document_id + position) as extra context "
    "(Candidate.parent_text) around each candidate - distinct from the sentence_window "
    "*chunker*, this works at retrieval time against whatever chunker produced the run's "
    "chunks. Matching stays precise on the small anchor chunk; the LLM sees the "
    "surrounding window. Run near the end of the pipeline, after filtering/reranking.",
)
class SentenceWindowExpander(PipelineStep):
    name = "sentence_window_expand"
    category = "expander"

    def __init__(self, window_size: int = 1):
        self.window_size = window_size  # neighbors on each side, by position

    def run(self, ctx: PipelineContext) -> PipelineContext:
        # Group all chunks by document, sorted by position, for neighbor lookup.
        by_document: dict[str, list] = {}
        for chunk in ctx.all_chunks:
            by_document.setdefault(chunk.document_id, []).append(chunk)
        for doc_chunks in by_document.values():
            doc_chunks.sort(key=lambda c: c.position)

        for candidate in ctx.candidates:
            doc_chunks = by_document.get(candidate.chunk.document_id, [])
            try:
                idx = next(i for i, c in enumerate(doc_chunks) if c.id == candidate.chunk.id)
            except StopIteration:
                continue
            lo = max(0, idx - self.window_size)
            hi = min(len(doc_chunks), idx + self.window_size + 1)
            window_chunks = doc_chunks[lo:hi]
            candidate.parent_text = " ".join(c.text for c in window_chunks)
        return ctx
