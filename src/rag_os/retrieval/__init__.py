"""
Importing this package registers every retrieval strategy - the LEGACY
single-retriever system (`retriever_registry`), used by runs created before
the Path-B pipeline refactor, and the new composable pipeline
(`pipeline_step_registry`) - and re-exports the public API both the pipeline
layer and the UI need. Both systems coexist; nothing here is mutually exclusive.
"""
from rag_os.retrieval import faiss_local, hybrid_bm25_vector, qdrant_cloud  # noqa: F401
from rag_os.retrieval.base import Retriever, retriever_registry
from rag_os.retrieval.pipeline_context import Candidate, PipelineContext
from rag_os.retrieval.pipeline_runner import PipelineStepConfig, run_pipeline
from rag_os.retrieval.pipeline_step import PipelineStep, pipeline_step_registry

__all__ = [
    "Retriever",
    "retriever_registry",
    "Candidate",
    "PipelineContext",
    "PipelineStep",
    "pipeline_step_registry",
    "PipelineStepConfig",
    "run_pipeline",
]
