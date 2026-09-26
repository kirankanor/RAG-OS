# retrieval/ — Path B pipeline refactor

## Drop-in instructions
Replace your existing `src/rag_os/retrieval/` folder with this one. Every
legacy file (`base.py`, `faiss_local.py`, `qdrant_cloud.py`,
`hybrid_bm25_vector.py`) is byte-for-byte what you already had (including the
RRF fusion_method added earlier) — old runs in your DB still load and work
unmodified via `run_manager.load_retriever_for_run()`.

Everything else here is new and additive: importing `rag_os.retrieval` now
also registers a second, parallel system (`pipeline_step_registry`) alongside
your existing `retriever_registry`. Nothing conflicts; both coexist.

## What's implemented (17 files, all syntax-checked)

| Category | File | Step name |
|---|---|---|
| core | `pipeline_context.py` | `Candidate`, `PipelineContext` |
| core | `pipeline_step.py` | `PipelineStep` ABC, `pipeline_step_registry` |
| core | `pipeline_runner.py` | `run_pipeline()`, `PipelineStepConfig` |
| generator | `generators/dense_vector.py` | `dense_flat`, `dense_hnsw` |
| generator | `generators/bm25.py` | `bm25` |
| generator | `generators/qdrant_dense.py` | `qdrant_dense` |
| fuser | `fusers/fuse.py` | `rrf_fuse`, `weighted_fuse` |
| filter | `filters/metadata_filter.py` | `metadata_filter` |
| expander | `expanders/expand.py` | `parent_document_expand`, `sentence_window_expand` |
| reranker | `rerankers/mmr.py` | `mmr` |
| reranker | `rerankers/existing_rerankers_adapter.py` | `cross_encoder_rerank`, `cohere_rerank` |

## How a pipeline is assembled (once wired into the UI — see below)

```python
from rag_os.retrieval import PipelineStepConfig, run_pipeline

steps = [
    PipelineStepConfig("dense_flat", {"fetch_k": 20}),
    PipelineStepConfig("bm25", {"fetch_k": 20}),
    PipelineStepConfig("rrf_fuse", {"k": 60}),
    PipelineStepConfig("metadata_filter", {"equals": {"source_filename": "handbook.pdf"}}),
    PipelineStepConfig("cross_encoder_rerank", {}),
    PipelineStepConfig("mmr", {"lambda_mult": 0.5}),
    PipelineStepConfig("parent_document_expand", {}),
]

results = run_pipeline(
    step_configs=steps,
    query=query_text,
    query_vector=query_embedding,
    all_chunks=chunks_for_run,          # list[Chunk]
    all_vectors=vector_by_chunk_id,     # dict[str, list[float]]
    top_k=5,
)
```

`results` is a `list[RetrievalResult]` — the same type your existing
`5_Retrieval.py` already renders, so display code needs no change once this
is wired in.

## NOT done yet (deliberately — needs your decisions first)

This folder is standalone and won't affect your running app until wired in.
Wiring means:

1. **`RunConfig`/`RunRow` schema change** — replace `retriever_name` +
   `retriever_params` (and `reranker_name`/`reranker_params`) with a single
   `pipeline_steps_json` column storing the ordered step list. Bigger than
   the reranker migration — needs its own `ALTER TABLE` + a decision on
   whether legacy single-retriever runs get auto-converted to a 1-step
   pipeline or just keep using the legacy path forever (recommended: keep
   legacy path for old runs, only new runs use the pipeline).
2. **`pipeline/dataset_generation.py`** — needs `all_vectors` built as a
   `dict[chunk_id -> vector]` (you currently pass parallel lists) and needs
   to persist the assembled step list against the run.
3. **`1_Upload_Documents.py`** — biggest UI change. Needs a step-builder:
   add step → pick category → pick strategy from
   `pipeline_step_registry.names()` filtered by category → params JSON →
   reorder. This replaces the current 4 fixed columns.
4. **`5_Retrieval.py`** — replace `retriever.retrieve()` +
   `reranker.rerank()` with one `run_pipeline()` call.
5. **Metadata schema decision** — `metadata_filter` works against whatever
   keys exist in `Chunk.metadata` today (e.g. `num_chars`, `num_pages` from
   parsers). Decide if you want richer, consistent metadata across all
   parsers (e.g. always include `source_filename`, `page_number`) for
   filtering to be genuinely useful.
6. **Parent-document chunking** — `parent_document_expand` is a no-op until
   some chunker actually sets `Chunk.parent_chunk_id`. No chunker does this
   yet — needs a two-tier chunking strategy (small chunks reference a larger
   parent chunk's id).

Ask for any of these six as a separate diff when you're ready — each is
scoped enough to do independently, in the order listed.
