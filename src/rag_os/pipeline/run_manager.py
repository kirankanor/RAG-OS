"""
Reload a previously-persisted run (documents/chunks/embeddings already in SQLite)
back into a live, queryable Retriever - so re-opening the Streamlit app doesn't
require re-parsing/re-chunking/re-embedding everything from scratch.
"""
from __future__ import annotations

from sqlmodel import select

from rag_os.core.types import Chunk
from rag_os.retrieval import retriever_registry
from rag_os.retrieval.base import Retriever
from rag_os.database.db import (
    ChunkRow,
    EmbeddingRow,
    RunRow,
    get_run,
    get_session,
    loads,
)
from rag_os.retrieval.rerankers import reranker_registry
from rag_os.retrieval.rerankers.base import Reranker



def load_retriever_for_run(run_id: str) -> tuple[Retriever, Reranker | None]:
    run = get_run(run_id)
    if run is None:
        raise ValueError(f"No run found with id '{run_id}'")

    with get_session() as session:
        chunk_rows = list(session.exec(select(ChunkRow).where(ChunkRow.run_id == run_id)))
        embedding_rows = list(
            session.exec(select(EmbeddingRow).where(EmbeddingRow.run_id == run_id))
        )

    embedding_by_chunk = {e.chunk_id: e for e in embedding_rows}

    chunks: list[Chunk] = []
    vectors: list[list[float]] = []
    for row in chunk_rows:
        chunks.append(
            Chunk(
                id=row.id,
                document_id=row.document_id,
                text=row.text,
                position=row.position,
                char_start=row.char_start,
                char_end=row.char_end,
                metadata=loads(row.metadata_json),
                chunker_name=row.chunker_name,
            )
        )
        emb = embedding_by_chunk.get(row.id)
        vectors.append(loads(emb.vector_json) if emb else [])

    retriever_params = loads(run.retriever_params)
    retriever = retriever_registry.create(run.retriever_name, **retriever_params)
    retriever.build(chunks, vectors)

    reranker = (
        reranker_registry.create(run.reranker_name, **loads(run.reranker_params))
        if run.reranker_name
        else None
    )
    return retriever, reranker


def delete_run(run_id: str) -> None:
    from rag_os.database.db import (
        ChunkRow,
        DocumentRow,
        EmbeddingRow,
        MetricRow,
        RatingRow,
    )

    with get_session() as session:
        for model in (ChunkRow, DocumentRow, EmbeddingRow, MetricRow, RatingRow, RunRow):
            rows = session.exec(select(model).where(model.run_id == run_id)) if model is not RunRow else None
            if model is RunRow:
                row = session.get(RunRow, run_id)
                if row:
                    session.delete(row)
                continue
            for row in rows:
                session.delete(row)
        session.commit()
