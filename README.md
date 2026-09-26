# RAG-OS

A modular monolith for experimenting with RAG pipeline strategies. Upload documents,
mix and match parsing / chunking / embedding / retrieval strategies, and compare the
results side by side in a local Streamlit UI.

## Architecture

Each pipeline stage lives in its own package under `src/rag_os/`, with one abstract
base interface and several interchangeable strategies that self-register into a
registry. No module imports its siblings directly — only `src/rag_os/pipeline/`
(the orchestration layer) and `app/` (the UI) are allowed to know about all of them.
That's what makes this a "modular monolith": one codebase and one deployable app,
but internally decoupled enough that adding a new chunker, say, never requires
touching the embedding or retrieval code.

```
src/rag_os/
  core/        shared dataclasses (Document, Chunk, EmbeddingRecord, RunConfig) + Registry
  parsing/     Parser interface + txt / pdf (x2) / docx / html strategies
  chunking/    Chunker interface + fixed_size / recursive_char / sentence_window /
               markdown_aware / semantic strategies
  embedding/   Embedder interface + local (sentence-transformers) / OpenAI / Cohere strategies
  retrieval/   Retriever interface + FAISS / Qdrant / hybrid BM25+vector strategies
  pipeline/    orchestrates parse -> chunk -> embed -> index for one "run", and persists it
  storage/     SQLite (via SQLModel) for run/document/chunk/embedding/rating rows,
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

# To use cloud strategies (OpenAI/Cohere embedders, Qdrant retriever):
uv sync --extra cloud

# Or both:
uv sync --extra local --extra cloud
```

If you'll use OpenAI, Cohere, or a remote Qdrant instance, copy `.env.example` to
`.env` and fill in the relevant keys:

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

## VS Code

Open this folder in VS Code — `.vscode/settings.json` points the Python interpreter
at `.venv` (created by `uv sync`) and enables Ruff as the formatter/linter on save.
Install the recommended extensions when prompted (Python, Pylance, Ruff).

`.vscode/launch.json` includes three debug configs (Run > Start Debugging, or the
Run and Debug panel):
- **Streamlit: Run RAG-OS App** — launches the app under the debugger so you can set
  breakpoints in any strategy file
- **Python: Current File** — runs whatever file is open
- **Python: Pytest (all tests)** — runs the test suite under the debugger

## Adding a new strategy

Every module follows the same pattern. To add, say, a new chunker:

1. Create `src/rag_os/chunking/my_chunker.py`, subclass `Chunker`, implement `chunk()`,
   and decorate the class with `@chunker_registry.register("my_chunker", "description")`.
2. Import it (for its registration side-effect) in `src/rag_os/chunking/__init__.py`.
3. It immediately shows up in the Upload Documents page's chunking strategy dropdown —
   no other code needs to change.

The same pattern applies to `parsing/`, `embedding/`, and `retrieval/`.

## Adding automatic metrics

`src/rag_os/evaluation/metrics.py` already implements `recall_at_k`, `precision_at_k`,
`mean_reciprocal_rank`, and `evaluate_run`. Once you have a set of
`{query: expected_chunk_ids}` pairs for your documents, call `evaluate_run(...)` from
the Reports page (or a new page) to get automatic scores per run, alongside the
manual thumbs-up/down ratings that already work today.
