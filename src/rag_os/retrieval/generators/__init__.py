"""
Importing this package registers every generator step with `pipeline_step_registry`.
Generators produce candidates from the full indexed chunk set (dense similarity,
lexical/BM25, or a hosted vector DB) - one or more can be combined in a pipeline,
merged by a fuser step afterward.
"""
from rag_os.retrieval.generators import bm25, dense_vector, qdrant_dense  # noqa: F401

__all__: list[str] = []
