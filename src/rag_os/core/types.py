"""
Shared data types passed between modules. Keeping these in `core` (instead of inside
each module) is what lets parsing/chunking/embedding/retrieval stay decoupled from
each other while still speaking a common language.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class Document:
    """Output of a parsing strategy: one uploaded file turned into clean text + metadata."""

    id: str = field(default_factory=_new_id)
    source_filename: str = ""
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    parser_name: str = ""
    created_at: str = field(default_factory=_utcnow)


@dataclass
class Chunk:
    """Output of a chunking strategy: one slice of a Document's text."""

    id: str = field(default_factory=_new_id)
    document_id: str = ""
    text: str = ""
    position: int = 0
    char_start: int = 0
    char_end: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    chunker_name: str = ""


@dataclass
class EmbeddingRecord:
    """Output of an embedding strategy: the vector for one chunk."""

    id: str = field(default_factory=_new_id)
    chunk_id: str = ""
    vector: list[float] = field(default_factory=list)
    dim: int = 0
    embedder_name: str = ""


@dataclass
class RetrievalResult:
    """One scored hit returned by a retrieval strategy for a given query."""

    chunk_id: str = ""
    score: float = 0.0
    rank: int = 0
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig:
    """
    One specific combination of strategies + params. This is what gets hashed/stored
    as a "run" so you can reproduce or compare it later.
    """

    id: str = field(default_factory=_new_id)
    name: str = ""
    parser_name: str = ""
    parser_params: dict[str, Any] = field(default_factory=dict)
    chunker_name: str = ""
    chunker_params: dict[str, Any] = field(default_factory=dict)
    embedder_name: str = ""
    embedder_params: dict[str, Any] = field(default_factory=dict)
    retriever_name: str = ""
    retriever_params: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utcnow)
    reranker_name: str = ""
    reranker_params: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utcnow)
