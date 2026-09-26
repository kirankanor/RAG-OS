"""
Importing this package registers every reranker step with `pipeline_step_registry`.
Rerankers reorder and trim candidates to ctx.top_k. Run these last in a pipeline
(after generators/fusers/filters), and before expanders if you want expansion to
apply only to the final chosen set.
"""
from rag_os.retrieval.rerankers import existing_rerankers_adapter, mmr  # noqa: F401

__all__: list[str] = []
