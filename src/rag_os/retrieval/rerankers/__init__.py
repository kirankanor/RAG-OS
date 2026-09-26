"""
Importing this package registers every reranking strategy: the LEGACY `Reranker`
interface (cross_encoder, cohere_rerank -> reranker_registry, used by old
retriever_name+reranker_name runs) AND the same strategies wrapped as
PipelineSteps for the new pipeline (mmr, cross_encoder_rerank, cohere_rerank ->
pipeline_step_registry). Both systems are re-exported from here since reranking/
was merged into retrieval/rerankers/ (previously a separate top-level package).
"""
from rag_os.retrieval.rerankers import cohere_rerank, cross_encoder  # noqa: F401
from rag_os.retrieval.rerankers import existing_rerankers_adapter, mmr  # noqa: F401
from rag_os.retrieval.rerankers.base import Reranker, reranker_registry

__all__ = ["Reranker", "reranker_registry"]
