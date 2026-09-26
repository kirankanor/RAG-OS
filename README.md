# RAG-OS

A modular monolith for experimenting with RAG pipeline strategies. Upload documents,
mix and match parsing / chunking / embedding / retrieval / reranking / generation
strategies, and compare the results side by side in a local Streamlit UI.

## Architecture

Each pipeline stage lives in its own package under `src/rag_os/`, with one abstract
base interface and several interchangeable strategies that self-register into a
registry. No module imports its siblings directly — only `src/rag_os/pipeline/`
(the orchestration layer) and `app/` (the UI) are allowed to know about all of them.

```
src/rag_os/
  core/        shared dataclasses (Document, Chunk, EmbeddingRecord, RunConfig) + Registry
  ingestion/
    parsing/     Parser interface + txt / pdf (x2) / docx / html strategies
    chunking/    Chunker interface + fixed_size / recursive_char / sentence_window /
                 markdown_aware / semantic / code_aware / contextual
    embedding/   Embedder interface + local (sentence-transformers) / OpenAI / Cohere
  retrieval/
    base.py, faiss_local.py, qdrant_cloud.py, hybrid_bm25_vector.py   -- legacy,
      single-retriever-per-run system, still used by runs created before the
      composable pipeline below existed
    pipeline_context.py / pipeline_step.py / pipeline_runner.py -- the composable
      pipeline: an ordered list of independent, user-toggleable PipelineSteps
    generators/  dense_flat, dense_hnsw, bm25, qdrant_dense -- produce candidates
    fusers/      rrf_fuse, weighted_fuse -- merge candidates from 2+ generators
    filters/     metadata_filter -- drop candidates by Chunk.metadata
    expanders/   parent_document_expand, sentence_window_expand -- attach extra context
    rerankers/   Reranker interface (legacy) + cross_encoder / cohere_rerank, PLUS
      the same strategies wrapped as PipelineSteps (mmr, cross_encoder_rerank,
      cohere_rerank) for the composable pipeline. Both systems live here together.
  generation/  Generator interface + groq_chat -- final answer synthesis from a
    query + retrieved chunks, via Groq's chat completions API
  pipeline/    orchestrates parse -> chunk -> embed -> index for one "run", and persists it
  database/    SQLite (via SQLModel) for run/document/chunk/embedding/rating rows,
               plus plain-file storage for uploaded files
  evaluation/  manual thumbs-up/down ratings (works now) + recall@k/precision@k/MRR
               metrics (wire in once you have ground-truth query/answer pairs)
  config/      settings (paths, API keys) via pydantic-settings, reads from .env

app/
  Home.py               landing page / instructions
  pages/1_Upload_Documents.py   upload files, pick strategies, run the pipeline
  pages/2_Parsing.py            inspect parsed text per run
  pages/3_Chunking.py           inspect chunks, sizes, boundaries
  pages/4_Embedding.py          inspect embedding dims + sample vectors
  pages/5_Retrieval.py          query playground + thumbs-up/down rating
  pages/6_Reports.py            compare runs, see rating summaries, delete runs
```

A "run" = one specific combination of strategies + params, applied to a batch of
uploaded files. Every run's documents, chunks, embeddings, and ratings are saved to
`data/rag_os.sqlite3`, so you can create as many runs as you want and compare them
later without re-processing anything.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install everything (base deps only — local/cloud strategies need the extras below)
uv sync

# To use local strategies (sentence-transformers embedder, FAISS/hybrid retrievers,
# PyMuPDF parser): 
uv sync --extra local

# To use cloud strategies (OpenAI/Cohere embedders, Qdrant retriever, Groq generator):
uv sync --extra cloud

# Or both:
uv sync --extra local --extra cloud
```

If you'll use OpenAI, Cohere, Groq, or a remote Qdrant instance, copy `.env.example`
to `.env` and fill in the relevant keys:

```bash
cp .env.example .env
```

## Running the app

```bash
uv run streamlit run app/Home.py
```

This opens the UI at `http://localhost:8501`. Start on **Upload Documents** to
create your first run.

## Running tests / linting

```bash
uv run pytest tests/ -v
uv run ruff check src/ app/
```

## Adding a new strategy

Every module follows the same pattern. To add, say, a new chunker:

1. Create `src/rag_os/ingestion/chunking/my_chunker.py`, subclass `Chunker`, implement
   `chunk()`, and decorate the class with `@chunker_registry.register("my_chunker", "description")`.
2. Import it (for its registration side-effect) in `src/rag_os/ingestion/chunking/__init__.py`.
3. It immediately shows up in the Upload Documents page's chunking strategy dropdown —
   no other code needs to change.

The same pattern applies to `ingestion/parsing/`, `ingestion/embedding/`, `retrieval/rerankers/`
(legacy `Reranker`) and `generation/` (`Generator`). For a new PipelineStep (generator,
fuser, filter, expander, or reranker for the composable retrieval pipeline), add the file
under the matching `retrieval/<category>/` folder, subclass `PipelineStep`, and register
with `pipeline_step_registry`.

## Retrieval: two systems, by design

`retrieval/` currently has two coexisting systems:
- **Legacy**: one `retriever_name` (+ optional `reranker_name`) per run — what every
  existing run in your database uses, loaded via `pipeline/run_manager.py`.
- **Composable pipeline**: an ordered list of `PipelineStepConfig`s run via
  `retrieval.run_pipeline(...)` — lets you mix multiple generators, a fuser, filters,
  expanders, and rerankers in one run instead of picking exactly one retriever.

Nothing forces a migration; both `retriever_registry` and `pipeline_step_registry`
are populated by importing `rag_os.retrieval`. Wiring the composable pipeline into
`RunConfig`/the Upload/Retrieval pages (so new runs can use it end-to-end) is not
done yet — see the project's handoff notes for the remaining steps.

## Adding automatic metrics

`src/rag_os/evaluation/metrics.py` already implements `recall_at_k`, `precision_at_k`,
`mean_reciprocal_rank`, and `evaluate_run`. Once you have a set of
`{query: expected_chunk_ids}` pairs for your documents, call `evaluate_run(...)` from
the Reports page (or a new page) to get automatic scores per run, alongside the
manual thumbs-up/down ratings that already work today.
