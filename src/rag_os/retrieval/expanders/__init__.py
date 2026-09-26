"""
Importing this package registers every expander step with `pipeline_step_registry`.
Expanders attach extra context to already-selected candidates (parent chunk, or
neighboring chunks) without changing which chunks matched.
"""
from rag_os.retrieval.expanders import expand  # noqa: F401

__all__: list[str] = []
